import os
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

def resolve_refs(schema, defs=None):
    """Recursively resolve $ref in schema using definitions."""
    if defs is None:
        defs = schema.get("$defs", {}) or schema.get("definitions", {})
    
    if isinstance(schema, dict):
        if "$ref" in schema:
            ref = schema["$ref"]
            name = ref.split("/")[-1]
            if name in defs:
                return resolve_refs(defs[name], defs)
        new_schema = {}
        for k, v in schema.items():
            if k in ["$defs", "definitions"]: continue
            new_schema[k] = resolve_refs(v, defs)
        return new_schema
    elif isinstance(schema, list):
        return [resolve_refs(item, defs) for item in schema]
    return schema

def clean_schema(schema):
    """
    Aggressively cleans Pydantic JSON schema for Gemini compatibility.
    Removes 'default', 'title', and handles 'anyOf' (Optional fields).
    """
    schema = resolve_refs(schema)
    
    def _clean(s):
        if isinstance(s, dict):
            # FIX: Handle Optional fields (anyOf: [type, null])
            if "anyOf" in s:
                for opt in s["anyOf"]:
                    if opt.get("type") != "null":
                        return _clean(opt) # Flatten 'anyOf' to the actual type
            
            # FIX: Remove forbidden keys
            for key in ["default", "title", "$defs", "definitions"]:
                if key in s:
                    del s[key]
            
            # Recurse
            for key, value in s.items():
                s[key] = _clean(value)
        
        elif isinstance(s, list):
            for i, item in enumerate(s):
                s[i] = _clean(item)
        return s

    return _clean(schema)

def get_generation_config(response_schema=None):
    config = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }
    if response_schema:
        try:
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                raw_schema = response_schema.model_json_schema()
                config["response_schema"] = clean_schema(raw_schema)
            else:
                config["response_schema"] = response_schema
        except Exception as e:
            print(f"Schema cleaning failed: {e}")
            # Fallback: try sending raw schema if cleaning fails
            config["response_schema"] = response_schema
            
    return config

# Verified Model Configuration
CONFIG = {
    "CLASSIFIER_MODEL": "gemini-2.5-flash",
    "EXTRACTOR_MODEL": "gemini-3-pro-preview"
}