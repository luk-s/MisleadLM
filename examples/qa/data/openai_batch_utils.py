from openai import OpenAI


def launch_batch_job(file_to_upload: str):
    client = OpenAI()

    print(f"Uploading batch file {file_to_upload}")
    batch_input_file = client.files.create(file=open(file_to_upload, "rb"), purpose="batch")

    print(f"Launching batch job for batch file {batch_input_file.id}")
    batch_input_file_id = batch_input_file.id

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )

    print(f"Batch job launched for batch file {batch_input_file_id}")


def retrieve_batch_results(file_to_download: str, result_file_name: str):
    client = OpenAI()

    # Download the results
    file_response = client.files.content(file_to_download)

    # Save the results to a file
    with open(result_file_name, "wb") as f:
        f.write(file_response.text.encode("utf-8"))

    print(f"Results saved to {result_file_name}")
