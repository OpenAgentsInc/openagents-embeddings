from sentence_transformers import SentenceTransformer
from openagents_grpc_proto import rpc_pb2_grpc
from openagents_grpc_proto import rpc_pb2
import time
import grpc
import json
import hashlib
import pickle
import os
import traceback
import base64
from sentence_transformers.quantization import quantize_embeddings

def log(rpcClient, message, jobId=None):
    print(message)
    if rpcClient and jobId: 
        rpcClient.logForJob(rpc_pb2.RpcJobLog(jobId=jobId, log=message))

class Vectorizer:
    
    def __init__(self,cache_path, device=-1):
        now = time.time()
        model = "intfloat/multilingual-e5-base"
        log(None, "Loading "+ model + " on device "+str(device))
        self.pipe = SentenceTransformer(model, device=device if device >= 0 else "cpu")
        log(None, "Model loaded in "+str(time.time()-now)+" seconds")
        self.cache_path=cache_path
        # Create cache directory
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

    def encode(self, sentences):       
        to_encode = []
        to_encode_index=[]
        out = []        
        for s in sentences:
            hash = hashlib.sha256(s.encode()).hexdigest()
            cache_file = self.cache_path+"/"+hash+".dat"
            if not os.path.exists(cache_file):
                to_encode.append(s)
                to_encode_index.append(len(out))
                out.append(None)
            else:
                with open(cache_file, "rb") as f:
                    out.append(pickle.load(f))

        encoded = self.pipe.encode(to_encode)
        for i in range(len(to_encode_index)):   
            out[to_encode_index[i]] = encoded[i]
            hash = hashlib.sha256(to_encode[i].encode()).hexdigest()
            with open(self.cache_path+"/"+hash+".dat", "wb") as f:
                pickle.dump(encoded[i], f)

        return out

    def quantize(self, embeddings):
        binary_embeddings = quantize_embeddings(embeddings, precision="binary")
        return binary_embeddings

    def split(self, text, chunk_size, overlap , marker,  out):
        tokens = self.pipe.tokenizer.tokenize(text)
        for i in range(0, len(tokens), chunk_size-overlap):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk = self.pipe.tokenizer.convert_tokens_to_string(chunk_tokens)
            out.append([chunk, marker])

    
def completePendingJob(rpcClient , act, CACHE_PATH):
    jobs=[]
    jobs.extend(rpcClient.getPendingJobs(rpc_pb2.RpcGetPendingJobs(filterByRunOn="openagents\\/embeddings")).jobs)    
    if len(jobs)>0 : log(rpcClient, str(len(jobs))+" pending jobs")
    for job in jobs:
        wasAccepted=False
        try:
            wasAccepted = True
            rpcClient.acceptJob(rpc_pb2.RpcAcceptJob(jobId=job.id))

            def getParamValue(key,default=None):
                param = [x for x in job.param if x.key == key]
                return param[0].value[0] if len(param) > 0 else default

            max_tokens = int(getParamValue("max-tokens", "1024"))
            overlap = int(getParamValue("overlap", "128"))
            quantize = getParamValue("quantize", "true") == "true"
            
            outputFormat = job.outputFormat

                    

            sentences = []
            for jin in job.input:
                data = jin.data
                data_type = jin.type 
                marker = jin.marker

                if data_type != "text":
                    # TODO: fetch?
                    raise Exception("Data type not supported")

                
                if marker != "query":
                    marker="passage"
                
                sentences.append([data,marker])

            blobDiskUrl = None
            blobRefId = None
            blobRefFileName = None
            blobDiskId = None
            blobCached = False
            if outputFormat=="application/hyperblob":
                blobRefId = str(max_tokens) + str(overlap) + str(quantize) + "".join([sentences[i][0] + ":" + sentences[i][1] for i in range(len(sentences))])
                blobRefId = hashlib.sha256(blobRefId.encode()).hexdigest()
                blobRefFileName = CACHE_PATH+"/"+blobRefId+".blob"
                if os.path.exists(blobRefFileName):
                    with open(blobRefFileName, "r") as f:
                        blobDiskUrl = f.read()
                    blobCached = True
                if not blobDiskUrl:
                    blobDiskUrl=rpcClient.createDisk(rpc_pb2.RpcCreateDiskRequest()).url
                blobDiskId=rpcClient.openDisk(rpc_pb2.RpcOpenDiskRequest(url=blobDiskUrl)).diskId

            if blobCached:
                rpcClient.completeJob(rpc_pb2.RpcJobOutput(jobId=job.id, output=blobDiskUrl))
            else:
                # Split long sentences
                sentences_chunks=[]
                for sentence in sentences:
                    act.split(sentence[0], max_tokens, overlap, sentence[1], sentences_chunks)
                sentences = sentences_chunks
                ##

                log(rpcClient,"Create embeddings for "+str(len(sentences))+" excerpts. max_tokens="+str(max_tokens)+", overlap="+str(overlap), job.id)

                t=time.time()
                embeddings = act.encode([sentences[i][1]+": "+sentences[i][0] for i in range(len(sentences))])
                if quantize:
                    embeddings = act.quantize(embeddings)

                log(rpcClient,"Embeddings created in "+str(time.time()-t)+" seconds", job.id)  


                if blobDiskId:
                    # write on disk
                    for i in range(len(sentences)):
                        dtype = embeddings[i].dtype
                        shape = embeddings[i].shape
                        embeddings_bytes =  embeddings[i].tobytes()
                        rpcClient.diskWriteSmallFile(rpc_pb2.RpcDiskWriteFileRequest(diskId=blobDiskId, path=str(i)+".embeddings.dtype", data=str(dtype).encode("utf-8")))
                        rpcClient.diskWriteSmallFile(rpc_pb2.RpcDiskWriteFileRequest(diskId=blobDiskId, path=str(i)+".embeddings.shape", data=json.dumps(shape).encode("utf-8")))
                        
                        CHUNK_SIZE = 1024 
                        def write_embeddings():
                            for j in range(0, len(embeddings_bytes), CHUNK_SIZE):
                                chunk = bytes(embeddings_bytes[j:j+CHUNK_SIZE])                                
                                request = rpc_pb2.RpcDiskWriteFileRequest(diskId=str(blobDiskId), path=str(i)+".embeddings.vectors", data=chunk)
                                yield request                              
                        rpcClient.diskWriteFile(write_embeddings())


                        sentences_bytes = sentences[i][0].encode("utf-8")
                        def write_sentences():
                            for j in range(0, len(sentences_bytes), CHUNK_SIZE):
                                chunk = bytes(sentences_bytes[j:j+CHUNK_SIZE])
                                request = rpc_pb2.RpcDiskWriteFileRequest(diskId=str(blobDiskId), path=str(i)+".embeddings", data=chunk)
                                yield request
                        rpcClient.diskWriteFile(write_sentences())

                    with open(blobRefFileName, "w") as f:
                        f.write(blobDiskUrl)            
                    
                    rpcClient.completeJob(rpc_pb2.RpcJobOutput(jobId=job.id, output=blobDiskUrl))

                else:
                    output = []
                    for i in range(len(sentences)):
                        dtype = embeddings[i].dtype
                        shape = embeddings[i].shape
                        embeddings_bytes =  embeddings[i].tobytes()
                        embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')                    
                        output.append(
                            [sentences[i][0], embeddings_b64, str(dtype), shape]
                        )



                    rpcClient.completeJob(rpc_pb2.RpcJobOutput(jobId=job.id, output=json.dumps(output)))

        except Exception as e:
            log(rpcClient, "Error processing job "+ str(e), job.id if job else None)
            if wasAccepted:
                rpcClient.cancelJob(rpc_pb2.RpcCancelJob(jobId=job.id, reason=str(e)))
            
            
            traceback.print_exc()



TEMPLATES = [
    {
        "nextAnnouncementTimestamp":0,
        "sockets": json.dumps({
            "in": {
                "max_tokens": {
                    "type": "number",
                    "value": 1000,
                    "description": "The maximum number of tokens for each text chunk",
                    "name": "Max Tokens"
                },
                "overlap": {
                    "type": "number",
                    "value": 128,
                    "description": "The number of tokens to overlap between each chunk",
                    "name": "Overlap"
                },
                "documents": {
                    "type": "array",
                    "description": "The documents to generate embeddings from",
                    "name": "Documents",
                    "schema": {
                        "data": {
                            "type": "string",
                            "description": "The data to generate embeddings from",
                            "name": "Data"
                        },
                        "data_type": {
                            "type": "string",
                            "value": "text",
                            "description": "The type of the data",
                            "name": "Data Type"
                        },
                        "marker": {
                            "type": "string",
                            "description": "'query' if it is a query or 'passage' if it is a passage",
                            "name": "Marker"
                        }
                    }
                }
            },
            "out": {
                "output": {
                    "type": "application/json",
                    "description": "The embeddings generated from the input data",
                    "name": "Embeddings"
                }
            }
        }),
        "meta":json.dumps({
            "kind": 5003,
            "name": "Embedding Generator Action",
            "about": "Generate embeddings from input documents",
            "tos": "",
            "privacy": "",
            "author": "",
            "web": "",
            "picture": "",
            "tags": ["tool"]
        }),
        "template":"""{
            "kind": {{meta.kind}},
            "created_at": {{sys.timestamp_seconds}},
            "tags": [
                ["output", "application/hyperblob"]
                ["param","run-on", "openagents/embeddings" ],                             
                ["param", "max-tokens", "{{in.max_tokens}}"],
                ["param", "overlap", "{{in.overlap}}"],
                ["param", "quantize", "{{in.quantize}}"],
                {{#in.documents}}
                ["i", "{{data}}", "{{data_type}}", "", "{{marker}}"],
                {{/in.documents}}
                ["expiration", "{{sys.expiration_timestamp_seconds}}"],
            ],
            "content":""
        }
        """
    }
]
NEXT_NODE_ANNOUNCE=0

def announce(rpcClient):    
    # Announce node
    global NEXT_NODE_ANNOUNCE
    time_ms=int(time.time()*1000)
    if time_ms >= NEXT_NODE_ANNOUNCE:
        ICON = os.getenv('NODE_ICON', "")
        NAME = os.getenv('NODE_NAME', "Embeddings Generator")
        DESCRIPTION = os.getenv('NODE_DESCRIPTION', "Generate embeddings for the input text")
        
        res=rpcClient.announceNode(rpc_pb2.RpcAnnounceNodeRequest(
            iconUrl = ICON,
            name = NAME,
            description = DESCRIPTION,
        ))
        NEXT_NODE_ANNOUNCE = int(time.time()*1000) + res.refreshInterval
    
    # Announce templates
    for template in TEMPLATES:
        if time_ms >= template["nextAnnouncementTimestamp"]:
            res = rpcClient.announceEventTemplate(rpc_pb2.RpcAnnounceTemplateRequest(
                meta=template["meta"],
                template=template["template"],
                sockets=template["sockets"]
            ))
            template["nextAnnouncementTimestamp"] = int(time.time()*1000) + res.refreshInterval



def main():
    DEVICE = int(os.getenv('TRANSFORMERS_DEVICE', "-1"))
    POOL_ADDRESS = os.getenv('POOL_ADDRESS', "127.0.0.1")
    POOL_PORT = int(os.getenv('POOL_PORT', "5000"))
    CACHE_PATH = os.getenv('CACHE_PATH', os.path.join(os.path.dirname(__file__), "cache"))
    t = Vectorizer(CACHE_PATH, DEVICE)
    while True:
        try:
            with grpc.insecure_channel(POOL_ADDRESS+":"+str(POOL_PORT)) as channel:
                stub = rpc_pb2_grpc.PoolConnectorStub(channel)
                log(stub, "Connected to "+POOL_ADDRESS+":"+str(POOL_PORT))
                while True:
                    try:
                        announce(stub)
                    except Exception as e:
                        log(stub, "Error announcing node "+ str(e), None)

                    try:
                        completePendingJob(stub, t, CACHE_PATH)
                    except Exception as e:
                        log(stub, "Error processing pending jobs "+ str(e), None)
                    time.sleep(10.0/1000.0)
        except Exception as e:
            log(None,"Error connecting to grpc server "+ str(e))
            
       

if __name__ == '__main__':
    main()