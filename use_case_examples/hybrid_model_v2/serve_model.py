"""Hybrid Model Deployment Server.

Routes:
    - Get all names
    - Get client.zip
    - Add a key
    - Compute
"""

import argparse
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

# No relative import here because when not used in the package itself
from concrete.ml.deployment import FHEModelServer
from concrete.ml.torch.hybrid_model import HybridFHEModelServer, underscore_str_to_tuple

if __name__ == "__main__":
    FILE_FOLDER = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", dest="port", type=int, default=8000)
    parser.add_argument(
        "--path-to-models",
        dest="path_to_models",
        type=Path,
        default=FILE_FOLDER / Path("compiled_models"),
    )
    parser.add_argument(
        "--path-to-keys",
        dest="path_to_keys",
        type=Path,
        default=FILE_FOLDER / Path("user_keys"),
    )
    args = parser.parse_args()
    app = FastAPI(debug=False)
    # Model-name -> Module-Name -> Input-shape
    server = HybridFHEModelServer(
        key_path=args.path_to_keys, model_dir=args.path_to_models, logger=logger
    )

    PORT = args.port

    upload_sessions = {}

    def check_inputs(
        server: HybridFHEModelServer,
        model_name: str,
        module_name: Optional[str],
        input_shape: Optional[Tuple],
    ):
        if model_name not in server.modules:
            raise HTTPException(
                status_code=500,
                detail=f"provided names '{model_name}' does not match any known name",
            )
        if module_name is not None and module_name not in server.modules[model_name]:
            raise HTTPException(
                status_code=500,
                detail=f"provided names '{module_name}' does not match any known name"
                f"{list(server.modules[model_name].keys())}",
            )
        if input_shape is not None and input_shape not in server.modules[model_name][module_name]:
            raise HTTPException(
                status_code=500,
                detail=f"provided names '{module_name}' does not match any known name"
                f"{list(server.modules[model_name][module_name].keys())}",
            )

    @app.get("/list_models")
    def list_models():
        return server.modules

    @app.get("/list_modules")
    def list_modules(model_name: str = Form()):
        check_inputs(server, model_name, None, None)
        return server.modules[model_name]

    @app.get("/list_shapes")
    def list_shapes(model_name: str = Form(), module_name: str = Form()):
        check_inputs(server, model_name, module_name, None)
        return server.modules[model_name][module_name]

    @app.get("/get_client")
    def get_client(model_name: str = Form(), module_name: str = Form(), input_shape: str = Form()):
        """Get client.

        Returns:
            FileResponse: client.zip

        Raises:
            HTTPException: if the file can't be find locally
        """
        check_inputs(server, model_name, module_name, input_shape)
        return FileResponse(
            server.get_client(model_name, module_name, input_shape), media_type="application/zip"
        )

    @app.post("/start_key_upload")
    async def start_key_upload(model_name: str = Form(...), module_name: str = Form(...), input_shape: str = Form(...)):
        """
        Start a key upload session. A temporary file is created to store incoming chunks.
        Returns an upload_id.
        """
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_file.close()  # We will open it for appending.
        upload_id = os.path.basename(tmp_file.name)
        upload_sessions[upload_id] = tmp_file.name
        logger.info(f"Started key upload session: {upload_id}")
        return {"upload_id": upload_id}

    @app.post("/upload_chunk")
    async def upload_chunk(upload_id: str = Form(...), chunk_index: int = Form(...), chunk: UploadFile = None):
        """
        Receive a single chunk and append it to the temporary file for the given upload_id.
        """
        if upload_id not in upload_sessions:
            raise HTTPException(status_code=400, detail="Invalid upload_id")
        file_path = upload_sessions[upload_id]
        chunk_data = await chunk.read()
        with open(file_path, "ab") as f:
            f.write(chunk_data)
        logger.info(f"Received chunk {chunk_index} for upload {upload_id}")
        return {"status": "chunk received", "chunk_index": chunk_index}

    @app.post("/finish_key_upload")
    async def finish_key_upload(upload_id: str = Form(...), model_name: str = Form(...), module_name: str = Form(...), input_shape: str = Form(...)):
        """
        Finalize the upload: read the complete file, remove the temporary file,
        pass the key data to the key-addition logic, and return a UID.
        """
        if upload_id not in upload_sessions:
            raise HTTPException(status_code=400, detail="Invalid upload_id")
        file_path = upload_sessions.pop(upload_id)
        with open(file_path, "rb") as f:
            key_data = f.read()
        os.remove(file_path)
        logger.info(f"Finished key upload for upload_id: {upload_id}")
        uid = server.add_key(key_data, model_name, module_name, input_shape)
        return {"uid": uid}

    def stream_response(
        encrypted_results: bytes, chunk_size: int = 1024 * 1024
    ) -> Generator[bytes, None, None]:
        """Yields chunks of encrypted results.

        Args:
            encrypted_results (bytes): The byte data to be streamed.
            chunk_size (int): The size of the chunks in which encrypted_results should be streamed.
                            Defaults to 1MB.

        Returns:
            bytes: Chunks of encrypted_results of size chunk_size.
        """
        buffer = io.BytesIO(encrypted_results)
        while chunk := buffer.read(chunk_size):
            yield chunk

    @app.post("/compute")
    async def compute(
        model_input: UploadFile,
        uid: str = Form(),
        model_name: str = Form(),
        module_name: str = Form(),
        input_shape: str = Form(),
    ):
        """
        Computes the circuit over encrypted input.

        Args:
            model_input (UploadFile): Input of the circuit.
            uid (str): The UID of the public key to use for computations.
            model_name (str): The name of the model to be used.
            module_name (str): The name of the module containing the computation circuit.
            input_shape (str): The shape of the input data.

        Returns:
            StreamingResponse: The result of the computation, streamed back in chunks.
        """
        check_inputs(server, model_name, module_name, input_shape)

        logger.info(f"UID for computation: {uid}")

        # Read the uploaded file first to avoid including this I/O time in FHE inference runtime measurement.
        logger.info("Reading uploaded data...")
        start_read = time.time()
        uploaded_data = await model_input.read()
        logger.info(f"Uploaded data read in {time.time() - start_read} seconds")
        encrypted_results = server.compute(
            uploaded_data,
            uid,
            model_name,
            module_name,
            input_shape,
        )
        return StreamingResponse(stream_response(encrypted_results))

    uvicorn.run(app, host="0.0.0.0", port=int(PORT))

