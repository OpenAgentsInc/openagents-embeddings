from OpenAgentsNode import OpenAgentsNode
from OpenAgentsNode import JobRunner
import config as NodeConfig
from events import embeddings as EmbeddingsEvent

from sentence_transformers import SentenceTransformer
import time
import grpc
import json
import hashlib
import pickle
import os
import traceback
import base64
from sentence_transformers.quantization import quantize_embeddings
import tiktoken
            
       
class Runner (JobRunner):
    def __init__(self, filters, meta, template, sockets):
        super().__init__(filters, meta, template, sockets)
        self.device = int(os.getenv('TRANSFORMERS_DEVICE', "-1"))
        self.cachePath = os.getenv('CACHE_PATH', os.path.join(os.path.dirname(__file__), "cache"))
        now = time.time()
        model = "intfloat/multilingual-e5-base"
        self.log("Loading "+ model + " on device "+str(self.device))
        self.pipe = SentenceTransformer(model, device=self.device if self.device >= 0 else "cpu")
        self.log( "Model loaded in "+str(time.time()-now)+" seconds")
        if not os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

    def split(self, text, chunk_size, overlap , marker,  out):
        enc = tiktoken.get_encoding("cl100k_base")
        tokenized_text = enc.encode(text)

        for i in range(0, len(tokenized_text), chunk_size-overlap):
            chunk_tokens = tokenized_text[i:min(i+chunk_size, len(tokenized_text))]
            chunk = enc.decode(chunk_tokens)
            out.append([chunk, marker])


      

        # tokens = self.pipe.tokenizer.tokenize(text)
        # for i in range(0, len(tokens), chunk_size-overlap):
        #     chunk_tokens = tokens[i:min(i+chunk_size, len(tokens))]
        #     chunk = self.pipe.tokenizer.convert_tokens_to_string(chunk_tokens)
        #     out.append([chunk, marker])

    def encode(self, sentences):       
        to_encode = []
        to_encode_index=[]
        out = []        
        for s in sentences:
            hash = hashlib.sha256(s.encode()).hexdigest()
            cache_file = self.cachePath+"/"+hash+".dat"
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
            with open(self.cachePath+"/"+hash+".dat", "wb") as f:
                pickle.dump(encoded[i], f)

        return out

    def quantize(self, embeddings):
        binary_embeddings = quantize_embeddings(embeddings, precision="binary")
        return binary_embeddings

    def run(self,job):
        def getParamValue(key,default=None):
            param = [x for x in job.param if x.key == key]
            return param[0].value[0] if len(param) > 0 else default

        # Extract parameters
        max_tokens = int(getParamValue("max-tokens", "1024"))
        overlap = int(getParamValue("overlap", "128"))
        quantize = getParamValue("quantize", "true") == "true"
        outputFormat = job.outputFormat

        # Extract input data
        sentences = []
        for jin in job.input:
            data = jin.data
            data_type = jin.type 
            marker = jin.marker
            if marker != "query": marker="passage"
            if data_type == "text":
                sentences.append([data,marker])
            elif data_type=="application/hyperblob":
                blobDisk = self.openStorage(data)
                files = blobDisk.list()
                supportedExts = ["html","txt","htm","md"]
                for file in [x for x in files if x.split(".")[-1] in supportedExts]:
                    tx=blobDisk.readUTF8(file)
                    sentences.append([tx,marker])
                blobDisk.close()
            else:
                raise Exception("Unsupported data type: "+data_type)

        # Check local cache
        cacheId = str(outputFormat) + str(max_tokens) + str(overlap) + str(quantize) + "".join([sentences[i][0] + ":" + sentences[i][1] for i in range(len(sentences))])
        cacheId = hashlib.sha256(cacheId.encode("utf-8")).hexdigest()
        cacheFile = os.path.join(self.cachePath, cacheId+".dat")
        if os.path.exists(cacheFile):
            with open(cacheFile, "r") as f:
                return f.read()
        
        # Split long sentences
        sentences_chunks=[]
        for sentence in sentences:
            self.split(sentence[0], max_tokens, overlap, sentence[1], sentences_chunks)
        sentences = sentences_chunks
        

        # Create embeddings
        self.log("Create embeddings for "+str(len(sentences))+" excerpts. max_tokens="+str(max_tokens)+", overlap="+str(overlap))
        embeddings = self.encode([sentences[i][1]+": "+sentences[i][0] for i in range(len(sentences))])
        if quantize:
            self.log("Quantize embeddings")
            embeddings = self.quantize(embeddings)


        # Serialize to an output format and return as string
        self.log("Embeddings ready. Serialize for output...")
        output = ""
        if outputFormat=="application/hyperblob":
            blobDisk = self.createStorage()
            for i in range(len(sentences)):
                dtype = embeddings[i].dtype
                shape = embeddings[i].shape
                sentences_bytes = sentences[i][0].encode("utf-8")
                embeddings_bytes =  embeddings[i].tobytes()
                blobDisk.writeBytes(str(i)+".embeddings.dtype", str(dtype).encode("utf-8"))
                blobDisk.writeBytes(str(i)+".embeddings.shape", json.dumps(shape).encode("utf-8"))
                blobDisk.writeBytes(str(i)+".embeddings", sentences_bytes)
                blobDisk.writeBytes(str(i)+".embeddings.vectors", embeddings_bytes)
                output = blobDisk.getUrl()
            blobDisk.close()
            with open(cacheFile, "w") as f:
                f.write(output)
        else:
            jsonOut = []
            for i in range(len(sentences)):
                dtype = embeddings[i].dtype
                shape = embeddings[i].shape
                embeddings_bytes =  embeddings[i].tobytes()
                embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')                    
                jsonOut.append(
                    [sentences[i][0], embeddings_b64, str(dtype), shape]
                )
            output=json.dumps(jsonOut)
            with open(cacheFile, "w") as f:
                f.write(output)

        return output

node = OpenAgentsNode(NodeConfig.meta)
node.registerRunner(Runner(filters=EmbeddingsEvent.filters,sockets=EmbeddingsEvent.sockets,meta=EmbeddingsEvent.meta,template=EmbeddingsEvent.template))
node.run()