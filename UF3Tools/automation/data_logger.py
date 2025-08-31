import os
import json
from datetime import datetime
import uuid
import io
import pickle
from collections.abc import Mapping 
import math

# Try to import libraries for both modes, but don't fail if one isn't installed.
try:
    import requests
except ImportError:
    requests = None

try:
    import pymongo
    from bson.objectid import ObjectId
    import gridfs
except ImportError:
    pymongo = None
    ObjectId = None
    gridfs = None


class DataLogger:
    """
    Logs data with a robust local file-based fallback system.
    Can operate in two modes:
    1. Direct to MongoDB (use_api=False): For secure environments like an HPC
       with an SSH tunnel. Requires `pymongo`.
    2. Via Web API (use_api=True): For connecting over the public internet.
       Requires `requests`.
    """
    def __init__(self, use_api=False, chunk_size_mb=4):
        self.use_api = use_api
        self.db_name = os.getenv("DB_NAME", "uf3tools")
        self.pending_uploads_dir = os.getenv("PENDING_UPLOADS_DIR", "pending_uploads")
        self.CHUNK_SIZE = chunk_size_mb * 1024 * 1024

        if self.use_api:
            if not requests:
                raise ImportError("The 'requests' library is required for API mode.")
            self.api_url = os.getenv("UF3TOOLS_API_URL")
            self.api_key = os.getenv("UF3TOOLS_API_KEY")
            if not self.api_url or not self.api_key:
                print("CRITICAL: In API mode, UF3TOOLS_API_URL and UF3TOOLS_API_KEY must be set.")
            self.json_headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
            self.stream_headers = {"X-API-Key": self.api_key, "Content-Type": "application/octet-stream"}
        else:
            if not pymongo or not gridfs:
                raise ImportError("The 'pymongo' library is required for Direct DB mode.")
            self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
            self.client = None
            self.db = None
            self.fs = None
            self.cert_file = os.getenv("MONGO_CERT_FILE")
            self.ca_file = os.getenv("MONGO_CA_FILE")    

        print(f"DataLogger initialized in {'API' if self.use_api else 'Direct DB'} mode.")
        os.makedirs(self.pending_uploads_dir, exist_ok=True)

    def is_connected(self):
        """Checks the connection status based on the current mode."""
        if self.use_api:
            try:
                test_url = f"{self.api_url}/{self.db_name}/studies?limit=1"
                response = requests.get(test_url, headers=self.json_headers, timeout=5)
                return response.status_code in [200, 404]
            except (requests.exceptions.RequestException, AttributeError):
                return False
        else:
            if self.client:
                try:
                    self.client.admin.command('ping')
                    return True
                except pymongo.errors.ConnectionFailure:
                    self.client = None
                    return False
            
            try:
                kwargs = {
                    "serverSelectionTimeoutMS": 5000
                }
                if self.cert_file and self.ca_file:
                    print("Attempting to connect to MongoDB using SSL/TLS certificates...")
                    kwargs['tls'] = True
                    kwargs['tlscertificatekeyfile'] = self.cert_file
                    kwargs['tlscafile'] = self.ca_file
                
                self.client = pymongo.MongoClient(self.mongo_uri, tlsAllowInvalidHostnames=True, **kwargs)
                self.client.admin.command('ping')
                self.db = self.client[self.db_name]
                self.fs = gridfs.GridFS(self.db)
                print("MongoDB connection successful.")
                return True
            except (pymongo.errors.ConnectionFailure, pymongo.errors.ServerSelectionTimeoutError, FileNotFoundError) as e:
                print(f"Failed to connect to MongoDB: {e}")
                self.client = None
                return False
                
    def log_document(self, collection_name, document, run_id):
        """Logs a small metadata document."""
        if "_id" not in document:
            document["_id"] = str(uuid.uuid4())
        document = self._sanitize_document(document)
        if self.is_connected():
            self._process_pending_uploads()
            try:
                if self.use_api:
                    self._log_document_api(collection_name, document)
                else:
                    self._log_document_db(collection_name, document)
                print(f"Successfully logged document to '{collection_name}'.")
            except Exception as e:
                print(f"Connection available, but write failed for '{collection_name}': {e}. Saving locally.")
                self._save_locally(document, run_id, collection_name)
        else:
            print(f"Connection not available. Saving document for '{collection_name}' locally.")
            self._save_locally(document, run_id, collection_name)

    def log_large_dataset(self, filename, dataset_obj, run_id):
        """Logs a large dataset file."""
        if self.is_connected():
            self._process_pending_uploads()
            try:
                if self.use_api:
                    return self._log_dataset_api(filename, dataset_obj, run_id)
                else:
                    return self._log_dataset_db(filename, dataset_obj, run_id)
            except Exception as e:
                print(f"Connection available, but file upload failed for '{filename}': {e}. Saving locally.")
                self._save_dataset_locally(filename, dataset_obj, run_id)
                return None
        else:
            print(f"Connection not available. Saving dataset '{filename}' locally.")
            self._save_dataset_locally(filename, dataset_obj, run_id)
            return None

    def _log_document_api(self, collection_name, document):
        sanitized = self._sanitize_document(document)
        url = f"{self.api_url}/{self.db_name}/{collection_name}"
        response = requests.post(url, headers=self.json_headers, data=json.dumps(sanitized))
        response.raise_for_status()

    def _log_document_db(self, collection_name, document):
        if "_id" in document and isinstance(document["_id"], str):
             try: document["_id"] = ObjectId(document["_id"])
             except Exception: pass 
        self.db[collection_name].replace_one({'_id': document['_id']}, document, upsert=True)

    def _log_dataset_api(self, filename, dataset_obj, run_id):
        init_url = f"{self.api_url}/fs/{self.db_name}/initiate"
        response = requests.post(init_url, headers=self.json_headers)
        response.raise_for_status()
        upload_id = response.json()['upload_id']
        
        data = pickle.dumps(dataset_obj)
        total_chunks = math.ceil(len(data) / self.CHUNK_SIZE)
        for i in range(total_chunks):
            chunk_data = data[i*self.CHUNK_SIZE : (i+1)*self.CHUNK_SIZE]
            chunk_url = f"{self.api_url}/fs/{self.db_name}/chunk?upload_id={upload_id}&index={i}"
            response = requests.post(chunk_url, headers=self.stream_headers, data=chunk_data)
            response.raise_for_status()
            
        finalize_url = f"{self.api_url}/fs/{self.db_name}/finalize?upload_id={upload_id}&filename={filename}&run_id={run_id}"
        response = requests.post(finalize_url, headers=self.json_headers)
        response.raise_for_status()
        file_id = response.json().get('file_id')
        print(f"Successfully uploaded dataset '{filename}' via API. File ID: {file_id}")
        return file_id

    def _log_dataset_db(self, filename, dataset_obj, run_id):
        metadata = {"run_id": run_id}
        existing = self.fs.find_one({"filename": filename, "metadata.run_id": run_id})
        if existing:
            self.fs.delete(existing._id)
        
        with self.fs.new_file(filename=filename, metadata=metadata) as f:
            pickle.dump(dataset_obj, f)
            file_id = f._id
        print(f"Successfully wrote dataset '{filename}' to GridFS. File ID: {file_id}")
        return file_id

    # --- Fallback and Recovery Logic ---
    
    def _save_locally(self, document, run_id, collection_name):
        run_dir = os.path.join(self.pending_uploads_dir, str(run_id))
        os.makedirs(run_dir, exist_ok=True)
        filepath = os.path.join(run_dir, f"{collection_name}_{uuid.uuid4()}.json")
        with open(filepath, 'w') as f:
            json.dump(document, f, indent=4)
        print(f"Saved document for '{collection_name}' to '{filepath}'.")

    def _save_dataset_locally(self, filename, dataset_obj, run_id):
        run_dir = os.path.join(self.pending_uploads_dir, str(run_id))
        os.makedirs(run_dir, exist_ok=True)
        filepath = os.path.join(run_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset_obj, f)
        print(f"Saved dataset '{filename}' locally to '{filepath}'.")

    def _process_pending_uploads(self):
        print("Checking for pending local files to upload...")
        for run_id in os.listdir(self.pending_uploads_dir):
            run_dir = os.path.join(self.pending_uploads_dir, run_id)
            if not os.path.isdir(run_dir): continue
            
            for filename in os.listdir(run_dir):
                filepath = os.path.join(run_dir, filename)
                try:
                    if filename.endswith(".json"):
                        collection_name = filename.split('_')[0]
                        with open(filepath, 'r') as f: document = json.load(f)
                        self.log_document(collection_name, document, run_id)
                    elif filename.endswith(".pkl"):
                        with open(filepath, 'rb') as f: dataset_obj = pickle.load(f)
                        self.log_large_dataset(filename, dataset_obj, run_id)
                    
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"Processed and removed pending file: {filename}")

                except Exception as e:
                    print(f"Failed to process pending file '{filename}': {e}. Will retry later.")
    
    def _sanitize_document(self, doc):
        """
        Recursively sanitizes a document for BSON (DB) and JSON (API/local) serialization.
        This version correctly handles any dictionary-like object, not just standard dicts.
        """
        # CHANGED: Use collections.abc.Mapping to correctly identify ANY dict-like object.
        if isinstance(doc, Mapping):
            return {str(key): self._sanitize_document(value) for key, value in doc.items()}
        
        # CHANGED: Check for sets and tuples in addition to lists.
        elif isinstance(doc, (list, tuple, set)):
            return [self._sanitize_document(item) for item in doc]
            
        elif isinstance(doc, (str, int, float, bool, type(None))):
            return doc
        else:
            # Fallback for any other type (custom objects, etc.) is to convert to string.
            return str(doc)
