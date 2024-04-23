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
from openai import OpenAI
import nlpcloud
import numpy as np
 
class Runner (JobRunner):
    openai = None
    nlpcloud = None

    def __init__(self, filters, meta, template, sockets):
        super().__init__(filters, meta, template, sockets)
        self.device = int(os.getenv('TRANSFORMERS_DEVICE', "-1"))
        self.cachePath = os.getenv('CACHE_PATH', os.path.join(os.path.dirname(__file__), "cache"))
        now = time.time()
        self.modelName = os.getenv('MODEL', "intfloat/multilingual-e5-base")
        self.maxTextLength = os.getenv('MAX_TEXT_LENGTH', 512)
        if self.modelName.startswith("nlpcloud:"):
            self.nlpcloud = nlpcloud.Client(self.modelName.replace("nlpcloud:",""), os.getenv('NLP_CLOUD_API_KEY'))
        elif self.modelName.startswith("openai:"):
            self.log("Using OpenAI API "+ self.modelName)
            self.openai = OpenAI()
            self.openaiModelName = self.modelName.replace("openai:","")
        else:
            self.log("Loading "+ self.modelName + " on device "+str(self.device))
            self.pipe = SentenceTransformer( self.modelName, device=self.device if self.device >= 0 else "cpu")
            self.log( "Model loaded in "+str(time.time()-now)+" seconds")
        self.addMarkersToSentences = os.getenv('ADD_MARKERS_TO_SENTENCES', "true") == "true"
        if not os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

 

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
            cached = await self.cacheGet(hash)
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
                encodedRaw=self.openai.embeddings.create(
                    input=[x[0] for x in to_encode],
                    model=self.openaiModelName
                )
                encoded = []
                for i in range(len(to_encode)):
                    embeddings = encodedRaw.data[i].embedding
                    encoded.append([np.array(embeddings),to_encode[i][1],to_encode[i][2]])          
            # TODO: more apis?
            else: # Use local models
                encodedRaw = self.pipe.encode([x[0] for x in to_encode], show_progress_bar=True)
                encoded = []
                for i in range(len(to_encode)):
                    embeddings = encodedRaw[i]
                    encoded.append([embeddings,to_encode[i][1],to_encode[i][2]])

            for i in range(len(encoded)):   
                embeddings = encoded[i][0]
                hash = encoded[i][1]
                index = encoded[i][2]
                out[index] = embeddings
                await  self.cacheSet(hash, embeddings)

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
            self.log("Use data: "+data)
            if marker != "query": marker="passage"
            if data_type == "text":
                sentences.append([data,marker])
            elif data_type=="application/hyperdrive+bundle":
                blobDisk = await  self.openStorage(data)
                files = await blobDisk.list()
                print("Found files",str(files))
                supportedExts = ["html","txt","htm","md"]
                for file in [x for x in files if x.split(".")[-1] in supportedExts]:
                    tx=await blobDisk.readUTF8(file)
                    sentences.append([tx,marker])
                await blobDisk.close()
            else:
                raise Exception("Unsupported data type: "+data_type)

        # Check local cache
        self.log("Check cache...")
        cacheId = hashlib.sha256(
            (str( self.modelName) + str(outputFormat) + str(max_tokens) + str(overlap) + str(quantize) + "".join([sentences[i][0] + ":" + sentences[i][1] for i in range(len(sentences))])).encode("utf-8")).hexdigest()
        cached = await self.cacheGet(cacheId)
        if cached is not None:
            self.log("Cache hit")
            return cached

        # Split long sentences
        self.log("Prepare sentences...")
        sentences_chunks=[]
        for sentence in sentences:
            self.prepare(sentence[0], max_tokens, overlap, sentence[1], sentences_chunks)
        sentences = sentences_chunks
        

        # Create embeddings
        self.log("Create embeddings for "+str(len(sentences))+" excerpts. max_tokens="+str(max_tokens)+", overlap="+str(overlap)+", quantize="+str(quantize)+", model="+str(self.modelName))
        embeddings =await self.encode([(sentences[i][1]+": "+sentences[i][0] if self.addMarkersToSentences  else sentences[i][0]) for i in range(len(sentences))])  
        if quantize:
            self.log("Quantize embeddings")
            embeddings = self.quantize(embeddings)

        # Serialize to an output format and return as string
        self.log("Embeddings ready. Serialize for output...")
        output = ""
        if outputFormat=="application/hyperdrive+bundle":
            blobDisk = await  self.createStorage()
            for i in range(len(sentences)):
                dtype = embeddings[i].dtype
                shape = embeddings[i].shape
                sentences_bytes = sentences[i][0].encode("utf-8")
                marker = sentences[i][1]
                embeddings_bytes =  embeddings[i].tobytes()
                await blobDisk.writeBytes(str(i)+".embeddings.dtype", str(dtype).encode("utf-8"))
                await blobDisk.writeBytes(str(i)+".embeddings.shape", json.dumps(shape).encode("utf-8"))
                await blobDisk.writeBytes(str(i)+".embeddings", sentences_bytes)
                await blobDisk.writeBytes(str(i)+".embeddings.kind", marker.encode("utf-8"))
                await blobDisk.writeBytes(str(i)+".embeddings.vectors", embeddings_bytes)
            output = blobDisk.getUrl()
            await blobDisk.close()
           
        else:
            jsonOut = []
            for i in range(len(sentences)):
                dtype = embeddings[i].dtype
                shape = embeddings[i].shape
                embeddings_bytes =  embeddings[i].tobytes()
                embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')                    
                jsonOut.append(
                    [sentences[i][0], embeddings_b64, str(dtype), shape , sentences[i][1]]
                )
            output=json.dumps(jsonOut)
          
        await  self.cacheSet(cacheId, output)
        return output

node = OpenAgentsNode(NodeConfig.meta)
node.registerRunner(Runner(filters=EmbeddingsEvent.filters,sockets=EmbeddingsEvent.sockets,meta=EmbeddingsEvent.meta,template=EmbeddingsEvent.template))
node.start()