filters = [
    {
        "filterByRunOn":"openagents\\/embeddings"
    }
]

sockets ={
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
        "model":{
            "type": "string",
            "value": "text",
            "description": "Specify which model to use. Empty for any",
            "name": "Model"        
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
}


meta = {
    "kind": 5003,
    "name": "Embedding Generator Action",
    "about": "Generate embeddings from input documents",
    "tos": "",
    "privacy": "",
    "author": "",
    "web": "",
    "picture": "",
    "tags": ["tool"]
}

template = """{
    "kind": {{meta.kind}},
    "created_at": {{sys.timestamp_seconds}},
    "tags": [
        ["output", "application/hyperdrive+bundle"]
        ["param","run-on", "openagents/embeddings" ],                             
        ["param", "max-tokens", "{{in.max_tokens}}"],
        ["param", "overlap", "{{in.overlap}}"],
        ["param", "quantize", "{{in.quantize}}"],
        ["param", "model", "{{in.model}}"],
        {{#in.documents}}
        ["i", "{{data}}", "{{data_type}}", "", "{{marker}}"],
        {{/in.documents}}
        ["expiration", "{{sys.expiration_timestamp_seconds}}"],
    ],
    "content":""
}
"""