from openagents import JobRunner,OpenAgentsNode,NodeConfig,RunnerConfig


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
from openai import OpenAI
import nlpcloud
import numpy as np
import asyncio
import concurrent
class EmbeddingsRunner (JobRunner):


    def __init__(self):
        super().__init__(
            RunnerConfig()\
                .kind(5003)\
                .name("Embedding Generator")\
                .description("Generate embeddings from input document")\
                .tos("https://openagents.com/terms") \
                .privacy("https://openagents.com/privacy")\
                .author("OpenAgentsInc")\
                .website("https://github.com/OpenAgentsInc/openagents-embeddings")\
                .picture("")\
                .tags([
                    "tool", 
                    "embeddings-generation"
                ]) \
                .filters()\
                    .filterByRunOn("openagents\\/embeddings") \
                    .commit()\
                .template("""{
                    "kind": {{meta.kind}},
                    "created_at": {{sys.timestamp_seconds}},
                    "tags": [
                        ["output", "application/hyperdrive+bundle"]
                        ["param","run-on", "openagents/embeddings" ],                             
                        ["param", "max-tokens", "{{in.max_tokens}}"],
                        ["param", "overlap", "{{in.overlap}}"],
                        ["param", "quantize", "{{in.quantize}}"],
                        ["param", "model", "{{in.model}}"],
                        ["output", "{{in.outputType}}"],
                        {{#in.documents}}
                        ["i", "{{data}}", "{{data_type}}", "", "{{marker}}"],
                        {{/in.documents}}
                        ["expiration", "{{sys.expiration_timestamp_seconds}}"],
                    ],
                    "content":""
                }
                """)\
                .inSocket("max_tokens","number")\
                    .description("The maximum number of tokens for each text chunk")\
                    .defaultValue(1000)\
                    .name("Max Tokens")\
                .commit()\
                .inSocket("overlap","number")\
                    .description("The number of tokens to overlap between each chunk")\
                    .defaultValue(128)\
                    .name("Overlap")\
                .commit()\
                .inSocket("model","string")\
                    .description("Specify which model to use. Empty for any")\
                    .defaultValue("")\
                    .name("Model")\
                .commit()\
                .inSocket("documents", "array")\
                    .description("The documents to generate embeddings from")\
                    .schema()\
                        .field("document", "map")\
                            .description("A document")\
                            .schema()\
                                .field("data", "string")\
                                    .description("The data to generate embeddings from")\
                                    .name("Data")\
                                .commit()\
                                .field("data_type", "string")\
                                    .description("The type of the data")\
                                    .defaultValue("text")\
                                    .name("Data Type")\
                                .commit()\
                                .field("marker", "string")\
                                    .description("'query' if it is a query or 'passage' if it is a passage")\
                                    .name("Marker")\
                                .commit()\
                            .commit()\
                        .commit()\
                    .commit()\
                .commit()\
                .inSocket("outputType", "string")\
                    .description("The Desired Output Type")\
                    .defaultValue("application/json")\
                .commit()\
                .outSocket("output", "string")\
                    .description("The embeddings generated from the input data")\
                    .name("Output")\
                .commit()\
            .commit()
        )


        self.openai = None
        self.nlpcloud = None
        self.device = int(os.getenv('EMBEDDINGS_TRANSFORMERS_DEVICE', "-1"))
        now = time.time()
        self.modelName =   os.getenv('EMBEDDINGS_MODEL', "intfloat/multilingual-e5-base")
        self.maxTextLength = int(os.getenv('EMBEDDINGS_MAX_TEXT_LENGTH', "512"))
        if self.modelName.startswith("nlpcloud:"):
            self.nlpcloud = nlpcloud.Client(self.modelName.replace("nlpcloud:",""), os.getenv('NLP_CLOUD_API_KEY'))
        elif self.modelName.startswith("openai:"):
            self.getLogger().info("Using OpenAI API "+ self.modelName)
            self.openai = OpenAI()
            self.openaiModelName = self.modelName.replace("openai:","")
        else:
            self.getLogger().info("Loading "+ self.modelName + " on device "+str(self.device))
            self.pipe = SentenceTransformer( self.modelName, device=self.device if self.device >= 0 else "cpu")
            self.getLogger().info( "Model loaded in "+str(time.time()-now)+" seconds")
        self.addMarkersToSentences = os.getenv('EMBEDDINGS_ADD_MARKERS_TO_SENTENCES', "true") == "true"


 

    def prepare(self, text, chunk_size, overlap , marker,  out):
        text = text.strip()
        enc = tiktoken.get_encoding("cl100k_base")
        tokenized_text = enc.encode(text)
        for i in range(0, len(tokenized_text), chunk_size-overlap):
            chunk_tokens = tokenized_text[i:min(i+chunk_size, len(tokenized_text))]
            chunk = enc.decode(chunk_tokens).strip()
            if len(chunk)>0:
                out.append([chunk, marker])
      

    async def encode(self, sentences):       


        out = []
        to_encode = []

        for s in sentences:
            hash = hashlib.sha256((self.modelName+":"+s).encode()).hexdigest()
            cached = await self.cacheGet(hash, local=True)
            if cached is not None:
                out.append(cached)
            else:
                to_encode.append([s,hash,len(out)])
                out.append(None)
        
        if len(to_encode)>0:
            encoded=None
            if self.nlpcloud :
                encodedRaw=self.nlpcloud.embeddings([x[0] for x in to_encode]).embeddings
                encoded = []
                for i in range(len(encodedRaw)):
                    embeddings = encodedRaw[i]
                    encoded.append([np.array(embeddings),to_encode[i][1],to_encode[i][2]])
            elif self.openai:
                CHUNK_SIZE = 1024
                encoded = []

                for i in range(0, len(to_encode), CHUNK_SIZE):
                    self.getLogger().log("Chunk",str(i))
                    chunk = to_encode[i:i+CHUNK_SIZE]
                    encodedRaw = self.openai.embeddings.create(
                        input=[x[0] for x in chunk],
                        model=self.openaiModelName
                    )
                    for j in range(len(chunk)):
                        embeddings = encodedRaw.data[j].embedding
                        encoded.append([np.array(embeddings), chunk[j][1], chunk[j][2]])

            # TODO: more apis?
            else: # Use local models
                encodedRaw = self.pipe.encode([x[0] for x in to_encode], show_progress_bar=True)
                encoded = []
                for i in range(len(to_encode)):
                    embeddings = encodedRaw[i]
                    encoded.append([embeddings,to_encode[i][1],to_encode[i][2]])

            waitList = []
            for i in range(len(encoded)):   
                embeddings = encoded[i][0]
                hash = encoded[i][1]
                index = encoded[i][2]
                out[index] = embeddings
                waitList.append(self.cacheSet(hash, embeddings, local=True))
            await asyncio.gather(*waitList)
        return out

    def quantize(self, embeddings):
        if len(embeddings) == 0:
            return embeddings
        binary_embeddings = quantize_embeddings(embeddings, precision="binary")
        return binary_embeddings

    async def canRun(self,job):
        def getParamValue(key,default=None):
            param = [x for x in job.param if x.key == key]
            return param[0].value[0] if len(param) > 0 else default
        model = getParamValue("model", self.modelName)
        return model == self.modelName

    async def run(self,job):
        def getParamValue(key,default=None):
            param = [x for x in job.param if x.key == key]
            return param[0].value[0] if len(param) > 0 else default

        # Extract parameters
        max_tokens = int(getParamValue("max-tokens", self.maxTextLength))
        max_tokens = min(max_tokens, self.maxTextLength)

        overlap = int(getParamValue("overlap",  int(max_tokens/3)))
        quantize = getParamValue("quantize", "true") == "true"
        outputFormat = job.outputFormat

        # Extract input data
        sentences = []
        for jin in job.input:
            data = jin.data
            data_type = jin.type 
            marker = jin.marker
            self.getLogger().log("Use data: "+data)
            if marker != "query": marker="passage"
            if data_type == "text":
                sentences.append([data,marker])
            elif data_type=="application/hyperdrive+bundle":
                blobDisk = await  self.openStorage(data)
                files = await blobDisk.list()
                self.getLogger().log("Found files",str(files))
                supportedExts = ["html","txt","htm","md"]
                for file in [x for x in files if x.split(".")[-1] in supportedExts]:
                    tx=await blobDisk.readUTF8(file)
                    sentences.append([tx,marker])
                await blobDisk.close()
            else:
                raise Exception("Unsupported data type: "+data_type)

        # Check local cache
        self.getLogger().info("Check cache...")
        cacheId = hashlib.sha256(
            (str( self.modelName) + str(outputFormat)
             + str(max_tokens) + str(overlap) + str(quantize) 
             + "".join([sentences[i][0] + ":" + sentences[i][1] for i in range(len(sentences))])).encode("utf-8")).hexdigest()
        self.getLogger().log("With cache id",cacheId)
        cached = await self.cacheGet(cacheId,local=True)
        if cached is not None:            
            return cached

        # Split long sentences
        self.getLogger().info("Prepare sentences...")
        sentences_chunks=[]
        for sentence in sentences:
            self.prepare(sentence[0], max_tokens, overlap, sentence[1], sentences_chunks)
        sentences = sentences_chunks
        

        # Create embeddings
        self.getLogger().info("Create embeddings for "+str(len(sentences))+" excerpts. max_tokens="+str(max_tokens)+", overlap="+str(overlap)+", quantize="+str(quantize)+", model="+str(self.modelName))
        embeddings =await self.encode([(sentences[i][1]+": "+sentences[i][0] if self.addMarkersToSentences  else sentences[i][0]) for i in range(len(sentences))])  
        if quantize:
            self.getLogger().info("Quantize embeddings")
            embeddings = self.quantize(embeddings)

        # Serialize to an output format and return as string
        self.getLogger().info("Embeddings ready. Serialize for output...")
        output = ""
        if outputFormat=="application/hyperdrive+bundle":
            blobDisk = await  self.createStorage()           
            
            sentencesOut = await blobDisk.openWriteStream("sentences.bin")
            await sentencesOut.writeInt(len(sentences))

            for i in range(len(sentences)):
                self.getLogger().log("Write sentence",str(i))
                sentence =  sentences[i][0].encode("utf-8")
                await sentencesOut.writeInt(len(sentence))
                await sentencesOut.write(sentence)

            embeddingsOut = await blobDisk.openWriteStream("embeddings.bin")
            await embeddingsOut.writeInt(len(embeddings))    
            for i in range(len(embeddings)):
                self.getLogger().log("Write embeddings",str(i))
                
                shape = embeddings[i].shape
                await embeddingsOut.writeInt(len(shape))
                for s in shape:
                    await embeddingsOut.writeInt(s)
                
                dtype = str(embeddings[i].dtype).encode()
                await embeddingsOut.writeInt(len(dtype))
                await embeddingsOut.write(dtype)

                bs = embeddings[i].tobytes()
                await embeddingsOut.writeInt(len(bs))
                await embeddingsOut.write(bs)

            await embeddingsOut.end()
            await sentencesOut.end()

            await embeddingsOut.close()
            await sentencesOut.close()
                

            output = blobDisk.getUrl()
            await blobDisk.close()

            
           
        else:

            jsonOut = []
            for i in range(len(sentences)):
                self.getLogger().log("Serialize embeddings",str(i))
                dtype = embeddings[i].dtype
                shape = embeddings[i].shape
                embeddings_bytes =  embeddings[i].tobytes()
                embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')                    
                jsonOut.append(
                    [sentences[i][0], embeddings_b64, str(dtype), shape , sentences[i][1]]
                )
            output=json.dumps(jsonOut)
          
        self.getLogger().info("Output ready. Cache and return.")
        await  self.cacheSet(cacheId, output,local=True)
        self.getLogger().log("Return output")
        return output

node = OpenAgentsNode(NodeConfig().name("Embeddings").description("Generate embeddings from input documents").version("1.0.0"))
node.registerRunner(EmbeddingsRunner())
node.start()