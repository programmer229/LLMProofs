import boto3
import uuid
import os
import json
from tqdm import tqdm
import sys
import threading

class TUMLClient:
    """
    This is a client to interact with the Tufa-Unified-Model-Library (TUML)
    TUML is an s3 bucket which is the source of truth for all custom models and checkpoints used at Tufa Labs.
    TUML has s3 versioning enabled, so multiple versions of the same checkpoint can exist, while old versions can be recovered.

    Features of the TUML client:
    - Upload checkpoints to TUML
    - Download checkpoints from TUML
    - List checkpoints, experiments, and projects in TUML
    - Query information about checkpoints in TUML
    - Delete checkpoints from TUML

    Checkpoints are downloaded to your machine's ~/.tuml_hub/ directory. The structure is as follows:
    ~/.tuml_hub/
        [project_name]/
            [experiment_name]/
                [checkpoint_name]/
                    config.json
                    generation_config.json
                    model-00001-of-00002.safetensors
                    model-00002-of-00002.safetensors
                    ... [any other files needed for the checkpoint]

    Make sure ~/.aws/credentials and ~/.aws/config are set up correctly.
    Logs or other metrics are not currentlystored in the TUML bucket. This may change in the future.
    """
    
    def __init__(self, bucket_name: str = "tufa-unified-model-library", library_file_path: str = "library_map.json"):
        """
        library_map.json is a json file that maps the project name, experiment name, and checkpoint name to a list of filenames.
        This is a helpful abstraction to have when uploading and downloading checkpoints, and querying for information.
        It functions as a lookup table, so we don't need to traverse the bucket, making almost all functions in this class O(1).

        library_map.json structure:
        {
            "[project_name]": {
                "[experiment_name]": {
                    "[checkpoint_name]": [
                        "[filename1_with_uuid]",
                        "[filename2_with_uuid]",
                        ...
                    ],
                    ...
                },
                ...
            },
            "[project_name]": {
                ...
            }
        }
        """

        self.bucket_name = bucket_name
        self.s3_resource = boto3.resource('s3')
        self.library_file_path = library_file_path
        
    def _load_library_map(self) -> dict:
        """
        Loads the library map from the TUML bucket.
        """
        library_map_obj = self.s3_resource.Object(self.bucket_name, self.library_file_path)
        library_map_utf8 = library_map_obj.get()['Body'].read().decode('utf-8')
        library_map = json.loads(library_map_utf8)

        return library_map

    def _write_library_map(self, library_map: dict) -> None:
        """
        Writes an updated library map to the TUML bucket.
        """
        library_map_obj = self.s3_resource.Object(self.bucket_name, self.library_file_path)
        library_map_obj.put(Body=json.dumps(library_map))

    def _s3_checkpoint_exists(self, project_name: str, experiment_name: str, checkpoint_name: str) -> bool:
        """
        Checks if a checkpoint exists in the TUML bucket.
        """
        library_map = self._load_library_map()

        try:
            library_map[project_name][experiment_name][checkpoint_name] is not None
            return True
        except KeyError:
            return False

    def _is_valid_checkpoint(self, path: str) -> bool:
        """
        Simple checks for if a directory is a valid HF model checkpoint. Returns True if is valid.
        """

        # If the path does not exist, return False.
        if not os.path.exists(path):
            return False
        
        # If the path is a file, return False.
        if os.path.isfile(path):
            return False
        
        # Check if there are no safetensors in the directory  (if there are none, it is not a valid checkpoint)
        if not any(filename.endswith(".safetensors") for filename in os.listdir(path)):
            return False
    
        # Check if there is a config.json file in the directory
        if not os.path.exists(os.path.join(path, "config.json")):
            return False
        
        return True

    def _prepare_filename(self, project_name: str, experiment_name: str, checkpoint_name: str, filename: str) -> str:
        """
        Prepares a filename for the TUML bucket.

        A prefix of 6 random chars is added to reduce latency. s3 assigns files to partitions based on prefix,
        so randomizing prefixes reduces the number of files partitioned together.
        """

        six_char_uuid = str(uuid.uuid4())[:6]
        filename = f"{six_char_uuid}/{project_name}/{experiment_name}/{checkpoint_name}/{filename}"
        return filename

    def _deconstruct_filename(self, filename: str) -> str:
        """
        Extracts the original filename from a TUML filename.
        """
        original_filename = filename.split("/")[-1]
        return original_filename

    def register_checkpoint(self, path: str, project_name: str, experiment_name: str, checkpoint_name: str, allow_overwrite: bool = False) -> None:
        """
        Registers a checkpoint in the TUML bucket.
        Inputs:
            path: str - The directory path to the checkpoint to register.
            project_name: str - The name of the project to register the checkpoint under.
            experiment_name: str - The name of the experiment to register the checkpoint under.
            checkpoint_name: str - The name of the checkpoint to register.
            allow_overwrite: bool - Whether to allow overwriting an existing checkpoint.
        Outputs:
            None
        """

        # Check if the path is a valid checkpoint
        if not self._is_valid_checkpoint(path):
            raise ValueError(f"Path {path} is not a valid HF model checkpoint.")

        # Check if the checkpoint already exists
        if self._s3_checkpoint_exists(project_name, experiment_name, checkpoint_name) and (not allow_overwrite):
            raise ValueError(f"Checkpoint {checkpoint_name} already exists in {project_name}/{experiment_name}. Set allow_overwrite=True to overwrite the current registered model checkpoint.")
        
        # Modify the library_map to include the new checkpoint
        library_map = self._load_library_map()
        if len(library_map) == 0:
            library_map[project_name] = {}
            library_map[project_name][experiment_name] = {}
        else:
            library_map[project_name] = library_map.get(project_name, {})
            library_map[project_name][experiment_name] = library_map[project_name].get(experiment_name, {})
        library_map[project_name][experiment_name][checkpoint_name] = []

        print("Beginning file uploads...")

        # For object in path to upload, generate the correct filename with uuid, add it to the library_map and upload the file
        for filename in tqdm(os.listdir(path), desc="Uploading files to TUML", unit="file"):
            prepared_filename = self._prepare_filename(project_name, experiment_name, checkpoint_name, filename)
            library_map[project_name][experiment_name][checkpoint_name].append(prepared_filename)

            # Upload the file to TUML
            self.s3_resource.Bucket(self.bucket_name).upload_file(
                Filename=os.path.join(path, filename),
                Key=prepared_filename,
                Callback=ProgressPercentage(os.path.join(path, filename))
            )

        # Write the updated library_map to the TUML bucket
        self._write_library_map(library_map)

        print(f"Uploaded and registered checkpoint {checkpoint_name} in {project_name}/{experiment_name} in the TUML bucket.")

    def download_registered_checkpoint(self, project_name: str, experiment_name: str, checkpoint_name: str):
        """
        Downloads a checkpoint from the TUML bucket to ~/.tuml_hub/
        Inputs:
            project_name: str - The name of the project to download the checkpoint from.
            experiment_name: str - The name of the experiment to download the checkpoint from.
            checkpoint_name: str - The name of the checkpoint to download.
        Outputs:
            None
        """
        # Check if the checkpoint exists
        if not self._s3_checkpoint_exists(project_name, experiment_name, checkpoint_name):
            raise ValueError(f"Checkpoint {checkpoint_name} does not exist at {project_name}/{experiment_name} in the TUML bucket.")

        # Create the ~/.tuml_hub/ directory if it doesn't exist
        download_directory = os.path.expanduser(f"~/.tuml_hub/{project_name}/{experiment_name}/{checkpoint_name}")
        os.makedirs(download_directory, exist_ok=True)

        # Download the checkpoint to ~/.tuml_hub/
        library_map = self._load_library_map()
        checkpoint_files = library_map[project_name][experiment_name][checkpoint_name]

        for filename in tqdm(checkpoint_files, desc="Downloading checkpoint files", unit="file"):
            original_filename = self._deconstruct_filename(filename)
            self.s3_resource.Bucket(self.bucket_name).download_file(
                Key=filename,
                Filename=os.path.join(download_directory, original_filename),
            )
        
        print(f"Downloaded checkpoint {checkpoint_name} from {project_name}/{experiment_name} to {download_directory}")

    def get_checkpoints(self, project_name: str, experiment_name: str):
        """
        Returns a list of all the checkpoints in an experiment.
        """
        library_map = self._load_library_map()
        checkpoints = list(library_map[project_name][experiment_name].keys())
        print(f"Checkpoints in {project_name}/{experiment_name}: {checkpoints}")
        return checkpoints

    def get_experiments(self, project_name: str):
        """
        Lists all the experiments in a project.
        """
        library_map = self._load_library_map()
        experiments = list(library_map[project_name].keys())
        return experiments

    def get_projects(self):
        """
        Lists all the projects in the TUML bucket.
        """
        library_map = self._load_library_map()
        projects = list(library_map.keys())
        return projects

    def get_library_map(self):
        """
        Returns the library map.
        """
        library_map = self._load_library_map()
        return library_map
    
    def reset_library_map(self):
        """
        Resets the library map to an empty dictionary.
        """
        self._write_library_map({})
    
    def get_statistics(self):
        """
        Returns statistics about the TUML bucket.
        """

        library_map = self._load_library_map()
        
        # Number of projects
        num_projects = len(library_map)
        print(f"Number of projects in TUML: {num_projects}")

        # Number of experiments
        num_experiments = sum(len(experiments) for experiments in library_map.values())
        print(f"Number of experiments in TUML: {num_experiments}")

class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

if __name__ == "__main__":
    client = TUMLClient()
    client.get_statistics()