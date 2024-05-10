from openagents import JobRunner,OpenAgentsNode,NodeConfig,RunnerConfig, Logger


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
            RunnerConfig(
                meta={
                    "kind": 5003,                    
                    "name": "Embedding Generator",
                    "description": "Generate embeddings from input document",
                    "tos": "https://openagents.com/terms",
                    "privacy": "https://openagents.com/privacy",
                    "author": "OpenAgentsInc",
                    "web": "https://github.com/OpenAgentsInc/openagents-embeddings",
                    "picture": "",
                    "tags": [ "embeddings-generation"],
                },
                filter={"filterByRunOn": "openagents\\/embeddings"},
                template="""{
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
                """,
                sockets={
                    "in":{
                        "max_tokens":{
                            "title":"Max Tokens",
                            "description":"The maximum number of tokens for each text chunk",
                            "type":"integer",
                            "default":1000
                        },
                        "overlap":{
                            "title":"Overlap",
                            "description":"The number of overlapping tokens between chunks",
                            "type":"integer",
                            "default":128
                        },
                        "model":{
                            "title":"Model",
                            "description":"The model to use for embeddings generation",
                            "type":"string"
                        },
                        "quantize":{
                            "title":"Quantize",
                            "description":"Quantize embeddings",
                            "type":"boolean",
                            "default":True
                        },
                        "documents":{
                            "title":"Documents",
                            "description":"The input documents",
                            "type":"array",
                            "items":{
                                "type":"object",
                                "properties":{
                                    "data":{
                                        "title":"Data",
                                        "description":"The input data",
                                        "type":"string",
                                        "format":"text"
                                    },
                                    "data_type":{
                                        "title":"Data Type",
                                        "description":"The type of the input data",
                                        "type":"string",
                                        "format":"text"
                                    },
                                    "marker":{
                                        "title":"Marker",
                                        "description":"The marker for the input data",
                                        "type":"string",
                                        "format":"text"
                                    }
                                }
                            }
                        },
                        "outputType":{
                            "title":"Output Type",
                            "description":"The output type",
                            "type":"string",
                            "default":"application/json"
                        }
                    },
                    "out":{
                        "content":{
                            "title":"Content",
                            "description":"The output content",
                            "type":"string"
                        }
                    }

                }
            )         
        )
        self.setRunInParallel(True)

    async def init (self,node):
        self.openai = None
        self.nlpcloud = None
        self.device = int(os.getenv('EMBEDDINGS_TRANSFORMERS_DEVICE', "-1"))
        now = time.time()
        self.modelName =   os.getenv('EMBEDDINGS_MODEL', "intfloat/multilingual-e5-base")
        self.maxTextLength = int(os.getenv('EMBEDDINGS_MAX_TEXT_LENGTH', "512"))
        
        logger=node.getLogger()

        if self.modelName.startswith("nlpcloud:"):
            self.nlpcloud = nlpcloud.Client(self.modelName.replace("nlpcloud:",""), os.getenv('NLP_CLOUD_API_KEY'))
        elif self.modelName.startswith("openai:"):
            logger.info("Using OpenAI API "+ self.modelName)
            self.openai = OpenAI()
            self.openaiModelName = self.modelName.replace("openai:","")
        else:
            logger.info("Loading "+ self.modelName + " on device "+str(self.device))
            self.pipe = SentenceTransformer( self.modelName, device=self.device if self.device >= 0 else "cpu")
            logger.info( "Model loaded in "+str(time.time()-now)+" seconds")
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
      

    async def encode(self, sentences,ctx):       
        logger=ctx.getLogger()

        out = []
        to_encode = []

        for s in sentences:
            hash = hashlib.sha256((self.modelName+":"+s).encode()).hexdigest()
            cached = await ctx.cacheGet(hash, local=True)
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
                    logger.log("Chunk",str(i))
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
                waitList.append(ctx.cacheSet(hash, embeddings, local=True))
            await asyncio.gather(*waitList)
        return out

    def quantize(self, embeddings):
        if len(embeddings) == 0:
            return embeddings
        binary_embeddings = quantize_embeddings(embeddings, precision="binary")
        return binary_embeddings

    async def canRun(self,ctx):
        model = ctx.getJobParamValue("model", self.modelName)
        return model == self.modelName

    async def run(self,ctx):
    
        # Extract parameters
        max_tokens = int(ctx.getJobParamValue("max-tokens", self.maxTextLength))
        max_tokens = min(max_tokens, self.maxTextLength)

        overlap = int(ctx.getJobParamValue("overlap",  int(max_tokens/3)))
        quantize = ctx.getJobParamValue("quantize", "true") == "true"
        outputFormat = ctx.getOutputFormat()

        job=ctx.getJob()
        logger=ctx.getLogger()
        
        # Extract input data
        sentences = []
        for jin in job.input:
            data = jin.data
            data_type = jin.type 
            marker = jin.marker
            
            if marker != "query": marker="passage"
            if data_type == "text":
                sentences.append([data,marker])
            elif data_type=="application/hyperdrive+bundle":
                disk = await  ctx.openStorage(data)
                files = await disk.list()
                logger.log("Found files",str(files))
                supportedExts = ["html","txt","htm","md"]
                for file in [x for x in files if x.split(".")[-1] in supportedExts]:
                    tx=await disk.readUTF8(file)
                    sentences.append([tx,marker])
                await disk.close()
            else:
                raise Exception("Unsupported data type: "+data_type)

        # Check local cache
        logger.info("Check cache...")
        cacheId = hashlib.sha256(
            (str( self.modelName) + str(outputFormat)
             + str(max_tokens) + str(overlap) + str(quantize) 
             + "".join([sentences[i][0] + ":" + sentences[i][1] for i in range(len(sentences))])).encode("utf-8")).hexdigest()
        logger.log("With cache id",cacheId)
        cached = await ctx.cacheGet(cacheId,local=True)
        if cached is not None:            
            return cached

        # Split long sentences
        logger.info("Prepare sentences...")
        sentences_chunks=[]
        for sentence in sentences:
            self.prepare(sentence[0], max_tokens, overlap, sentence[1], sentences_chunks)
        sentences = sentences_chunks
        

        # Create embeddings
        logger.info("Create embeddings for "+str(len(sentences))+" excerpts. max_tokens="+str(max_tokens)+", overlap="+str(overlap)+", quantize="+str(quantize)+", model="+str(self.modelName))
        embeddings =await self.encode([(sentences[i][1]+": "+sentences[i][0] if self.addMarkersToSentences  else sentences[i][0]) for i in range(len(sentences))],ctx)  
        if quantize:
            logger.info("Quantize embeddings")
            embeddings = self.quantize(embeddings)

        # Serialize to an output format and return as string
        logger.info("Embeddings ready. Serialize for output...")
        output = ""
        if outputFormat=="application/hyperdrive+bundle":
            disk = await  ctx.createStorage()           
            
            sentencesOut = await disk.openWriteStream("sentences.bin")
            await sentencesOut.writeInt(len(sentences))

            for i in range(len(sentences)):
                logger.log("Write sentence",str(i))
                sentence =  sentences[i][0].encode("utf-8")
                await sentencesOut.writeInt(len(sentence))
                await sentencesOut.write(sentence)

            embeddingsOut = await disk.openWriteStream("embeddings.bin")
            await embeddingsOut.writeInt(len(embeddings))    
            for i in range(len(embeddings)):
                logger.log("Write embeddings",str(i))
                
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
                

            output = disk.getUrl()
            await disk.close()

            
           
        else:

            jsonOut = []
            for i in range(len(sentences)):
                logger.log("Serialize embeddings",str(i))
                dtype = embeddings[i].dtype
                shape = embeddings[i].shape
                embeddings_bytes =  embeddings[i].tobytes()
                embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')                    
                jsonOut.append(
                    [sentences[i][0], embeddings_b64, str(dtype), shape , sentences[i][1]]
                )
            output=json.dumps(jsonOut)
          
        logger.info("Output ready. Cache and return.")
        await  ctx.cacheSet(cacheId, output,local=True)
        logger.log("Return output")
        return output

node = OpenAgentsNode(NodeConfig({
    "name": "OpenAgents Embeddings",
    "description": "Embeddings generation service",
    "version": "0.1.0",
}))
node.registerRunner(EmbeddingsRunner())
node.start()