import os
import json
from datetime import datetime
import uuid
import io
import pickle
import gridfs
from collections.abc import Mapping 
import math
from UF3Tools.automation.ModelAutomater import *

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


class DatabaseReader:
    def __init__(self, use_api=False):
        self.use_api = use_api
        self.db_name = os.getenv("DB_NAME", "uf3tools")
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

        print(f"DatabaseReader initialized in {'API' if self.use_api else 'Direct DB'} mode.")
        
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
            
    def get_potential_data(self, potential_id):
        """Fetches a potential file from the database by its ID."""
        if not self.is_connected():
            print("Not connected to the database.")
            return None
        
        try:
            if self.use_api:
                raise NotImplementedError("API mode for fetching potential data is not implemented.")
            else:
                database_entry = self.db['trials'].find_one({"_id": (potential_id)})
                if not database_entry:
                    print(f"No trial found with ID '{potential_id}'.")
                    return None
                else:
                    return database_entry
        except Exception as e:
            print(f"Failed to retrieve potential with ID '{potential_id}': {e}")
            return None
    
    def get_study_results(self, study_id):
        """Fetches all trials associated with a given study ID."""
        if not self.is_connected():
            print("Not connected to the database.")
            return []
        
        try:
            if self.use_api:
                raise NotImplementedError("API mode for fetching study results is not implemented.")
            else:
                trials = list(self.db['trials'].find({"study_id": study_id}))
                return trials
        except Exception as e:
            print(f"Failed to retrieve trials for study ID '{study_id}': {e}")
            return []
        
    def get_all_studies(self):
        """Fetches all studies from the database."""
        if not self.is_connected():
            print("Not connected to the database.")
            return []
        try:
            if self.use_api:
                raise NotImplementedError("API mode for fetching all studies is not implemented.")
            else:
                studies = list(self.db['studies'].find({}))
                return studies
        except Exception as e:
            print(f"Failed to retrieve studies: {e}")
            return []
        
    def get_configuration(self, study_id):
        """Fetches the configuration document for a given study ID."""
        if not self.is_connected():
            print("Not connected to the database.")
            return None
        
        try:
            if self.use_api:
                raise NotImplementedError("API mode for fetching configuration is not implemented.")
            else:
                config = self.db['configs'].find_one({"_id": study_id})
                if not config:
                    print(f"No configuration found for study ID '{study_id}'.")
                    return None
                else:
                    return config
        except Exception as e:
            print(f"Failed to retrieve configuration for study ID '{study_id}': {e}")
            return None
        
class TrialReader:
    def __init__(self, trial):
        self.trial = trial
    def get_params(self):
        return self.trial.get('params', {})
    def get_loss(self):
        return self.trial.get('value', None)
    def get_all_validation_data(self):
        return self.trial.get('validation_data', None)
    def get_structure_validation_data(self, structure_id):
        validation_data = self.get_all_validation_data()
        if not validation_data:
            return None
        if isinstance(validation_data, list):
            for data in validation_data:
                if data.get('structure') == structure_id:
                    return data.get('validation_data', None)
        elif isinstance(validation_data, dict):
            if validation_data.get('structure') == structure_id:
                return validation_data.get('validation_data', None)
        return None
    
 
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
                    self._log_document_api(collection_name, document, run_id)
                else:
                    self._log_document_db(collection_name, document, run_id)
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

    def _log_document_api(self, collection_name, document, run_id=None):
        try:
            sanitized = self._sanitize_document(document)
            url = f"{self.api_url}/{self.db_name}/{collection_name}"
            response = requests.post(url, headers=self.json_headers, data=json.dumps(sanitized))
            response.raise_for_status()
        except:
            return

    def _log_document_db(self, collection_name, document, run_id=None):
        try:
            if "_id" in document and isinstance(document["_id"], str):
                try: document["_id"] = ObjectId(document["_id"])
                except Exception: pass 
            self.db[collection_name].replace_one({'_id': document['_id']}, document, upsert=True)
        except:
            return
    def log_file(self, filename, file_data, run_id, content_type='application/octet-stream'):
        """Logs a generic file (binary or text) to GridFS."""
        if self.is_connected():
            self._process_pending_uploads()
            try:
                if self.use_api:
                    # This would require API endpoint changes; focusing on direct DB mode
                    raise NotImplementedError("log_file via API is not yet implemented.")
                else:
                    return self._log_file_db(filename, file_data, run_id, content_type)
            except Exception as e:
                print(f"Connection available, but file upload failed for '{filename}': {e}. Saving locally.")
                self._save_file_locally(filename, file_data, run_id)
                return None
        else:
            print(f"Connection not available. Saving file '{filename}' locally.")
            self._save_file_locally(filename, file_data, run_id)
            return None

    def _log_file_db(self, filename, file_data, run_id, content_type):
        """Helper to log a generic file to GridFS."""
        metadata = {"run_id": run_id, "content_type": content_type}
        existing = self.fs.find_one({"filename": filename, "metadata.run_id": run_id})
        if existing:
            self.fs.delete(existing._id)
        
        file_id = self.fs.put(file_data, filename=filename, metadata=metadata)
        print(f"Successfully wrote file '{filename}' to GridFS. File ID: {file_id}")
        return file_id

    def _save_file_locally(self, filename, file_data, run_id):
        """Saves a generic file locally."""
        run_dir = os.path.join(self.pending_uploads_dir, str(run_id))
        os.makedirs(run_dir, exist_ok=True)
        filepath = os.path.join(run_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(file_data)
        print(f"Saved file '{filename}' locally to '{filepath}'.")
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
        """Saves a document locally in the pending uploads directory."""
        run_dir = os.path.join(self.pending_uploads_dir, str(run_id))
        os.makedirs(run_dir, exist_ok=True)
        
        doc_id = document.get("_id", uuid.uuid4())
        filename = f"{collection_name}_{doc_id}.json"
        filepath = os.path.join(run_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(document, f, indent=4)
        print(f"Saved document for '{collection_name}' to '{filepath}'.")

    def _save_dataset_locally(self, filename, dataset_obj, run_id):
        """Saves a dataset file locally."""
        run_dir = os.path.join(self.pending_uploads_dir, str(run_id))
        os.makedirs(run_dir, exist_ok=True)
        
        filepath = os.path.join(run_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset_obj, f)
        print(f"Saved dataset '{filename}' locally to '{filepath}'.")

    def _process_pending_uploads(self):
        """
        Processes and uploads any locally saved files.
        This version is corrected to avoid recursion and handle files correctly.
        """
        if not os.path.exists(self.pending_uploads_dir):
            return

        print("Checking for pending local files to upload...")
        for run_id in os.listdir(self.pending_uploads_dir):
            run_dir = os.path.join(self.pending_uploads_dir, run_id)
            
            if not os.path.isdir(run_dir):
                continue
            
            files_to_process = os.listdir(run_dir)
            if not files_to_process:
                os.rmdir(run_dir)
                continue

            print(f"--- Processing pending run: {run_id} ---")
            for filename in files_to_process:
                filepath = os.path.join(run_dir, filename)
                
                if not os.path.isfile(filepath):
                    continue

                try:
                    upload_successful = False
                    if filename.endswith(".json"):
                        collection_name = filename.split('_')[0]
                        with open(filepath, 'r') as f:
                            document = json.load(f)
                        
                        if self.use_api:
                            self._log_document_api(collection_name, document)
                        else:
                            self._log_document_db(collection_name, document)
                        upload_successful = True

                    elif filename.endswith(".pkl"):
                        with open(filepath, 'rb') as f:
                            dataset_obj = pickle.load(f)
                        
                        
                        if self.use_api:
                            self._log_dataset_api(filename, dataset_obj, run_id)
                        else:
                            self._log_dataset_db(filename, dataset_obj, run_id)
                        upload_successful = True
                    
                    if upload_successful:
                        os.remove(filepath)
                        print(f"Processed and removed pending file: {filename}")

                except Exception as e:
                    print(f"Failed to process pending file '{filename}': {e}. Will retry later.")
    
    def _sanitize_document(self, doc):
        """
        Recursively sanitizes a document for BSON (DB) and JSON (API/local) serialization.
        This version correctly handles any dictionary-like object, not just standard dicts.
        """

        if isinstance(doc, Mapping):
            return {str(key): self._sanitize_document(value) for key, value in doc.items()}
        
        elif isinstance(doc, (list, tuple, set)):
            return [self._sanitize_document(item) for item in doc]
            
        elif isinstance(doc, (str, int, float, bool, type(None))):
            return doc
        else:
            return str(doc)



class LoggingSession:
    """Helps you log to the database from single shot runs 
    """
    def __init__(self, automater, study_name=None):
        self.logger = DataLogger()
        self.automater = automater
        self.run_id = f'{study_name}_{uuid.uuid4().hex[:32]}' if study_name is not None else f'{uuid.uuid4().hex[:32]}'
        self.study_name = study_name if study_name is not None else self.run_id,
        self.study_doc = {
            "_id": self.run_id,
            "study_name": self.study_name,
            "chemical_system": automater.get_chemical_system().as_dict(),
            "status": "RUNNING",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "best_value": None,
            "best_params": None
        }
        print(f'--- Logging session started with Run ID: {self.run_id} ---')
    def serialize_validation_data(self, validation_data):
        """
        Recursively converts a ValidationData object or list of objects
        into a JSON-serializable dictionary or list of dictionaries.
        """
        if isinstance(validation_data, list):
            return [self.serialize_validation_data(data) for data in validation_data if data is not None]
        elif isinstance(validation_data, ValidationData):
            return validation_data.to_mongo_dict()
        else:
            return None
    def log_study(self):
        self.logger.log_document("studies", self.study_doc, self.run_id)
    def log_config(self, settings_dict={}):
        config_doc = {
            "_id": self.run_id, 
            "settings": settings_dict, 
            "chemical_system": self.automater.get_chemical_system().as_dict(),
            "bspline_config": self.automater.get_bspline_config().as_dict(),
            "data_sources": self.automater.get_all_filtered_labels(),
            "loss_weights": settings_dict.get('loss_weights', None),
            "acceptable_errors": settings_dict.get('acceptable_errors', None),
            "initial_sample_weights": settings_dict.get('sample_weights', None),
            "validation_structures": settings_dict.get('testing_mp_structures', None),
            "validations_to_run": settings_dict.get('validations_to_run', None),
            "filter_history": self.automater.get_filter_history()
        }
        self.logger.log_document("configs", config_doc, self.run_id)
        
    def log_datasets(self,):
        dataset_doc = {
            "_id": self.run_id,
            "training_keys": self.automater.get_training_keys(),
            "testing_keys": self.automater.get_testing_keys()
        }
        self.logger.log_document("datasets", dataset_doc, self.run_id)
        
    def log_trial(self, potential_name, trial_number, params, loss, validation_data,visualization_files={}):
        serialized_validation = self.serialize_validation_data(validation_data)
        trial_id = os.path.basename(potential_name)
        trial_doc = {
            "_id": trial_id,
            "file_path": potential_name,
            "study_id": self.run_id,
            "trial_number": trial_number,
            "params":  params,
            "value": loss,
            "validation_data": serialized_validation,
            "status": "COMPLETED",
            "timestamp": datetime.now().isoformat()
        }
        visualization_ids = {}
        if visualization_files:
            for filename, file_data in visualization_files.items():
                file_id = self.logger.log_file(filename, file_data, self.run_id, content_type='image/png')
                if file_id:
                    visualization_ids[filename] = str(file_id)

        trial_doc["visualization_ids"] = visualization_ids
        
        self.logger.log_document("trials", trial_doc, self.run_id)
        
        #update the best values
        if (self.study_doc["best_value"] is None) or (loss < self.study_doc["best_value"]):
            self.study_doc["best_value"] = loss
            self.study_doc["best_params"] = params,
            self.logger.log_document("studies", self.study_doc, self.run_id)
            
    def log_document(self, collection_name, document, run_id):
        self.logger.log_document(collection_name, document, run_id)
            
        