import os
import shutil
import subprocess
import tempfile
import pytube
import whisper
import openai
import pytesseract
from PIL import Image
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from neo4j import GraphDatabase

# Set OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Neo4j connection details
neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
neo4j_password = os.environ.get("NEO4J_PASSWORD", "your_password")

driver = get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_password)

def download_youtube_video(url, output_path):
    yt = pytube.YouTube(url)
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video_file = video_stream.download(output_path=output_path)
    video_title = yt.title
    return video_file, video_title

def extract_audio(video_file, audio_file):
    command = f"ffmpeg -i \"{video_file}\" -q:a 0 -map a \"{audio_file}\" -y"
    subprocess.run(command, shell=True)
    return audio_file

def transcribe_audio_with_timestamps(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, verbose=False)
    return result['segments']

def extract_screenshot(video_file, time_in_seconds, output_image_file):
    command = f"ffmpeg -ss {time_in_seconds} -i \"{video_file}\" -frames:v 1 \"{output_image_file}\" -y"
    subprocess.run(command, shell=True)
    return output_image_file

def extract_text_from_image(image_file):
    text = pytesseract.image_to_string(Image.open(image_file))
    return text

def summarize_image(image_file):
    # Load BLIP model and processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)

    image = Image.open(image_file).convert('RGB')
    inputs = processor(image, return_tensors='pt').to(device)
    out = model.generate(**inputs)
    summary = processor.decode(out[0], skip_special_tokens=True)
    return summary

def extract_security_info(text):
    prompt = f"""
    Extract any security controls and the security risks they mitigate from the following text.
    Provide the information in the following JSON format:
    {{
        "security_controls": [{{"control": "...", "risk_mitigated": "..."}}, ...]
    }}
    Text:
    {text}
    """
    response = openai.Completion.create(
        engine='gpt-3.5-turbo',
        prompt=prompt,
        max_tokens=500,
        temperature=0,
        n=1,
        stop=None
    )
    try:
        extracted_info = eval(response.choices[0].text.strip())
        return extracted_info.get('security_controls', [])
    except:
        return []

def get_neo4j_driver(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

def store_security_info_in_neo4j(driver, security_info, video_title, video_url):
    with driver.session() as session:
        for item in security_info:
            control = item.get('control')
            risk = item.get('risk_mitigated')
            if control and risk:
                session.run("""
                    MERGE (c:SecurityControl {name: $control})
                    MERGE (r:SecurityRisk {description: $risk})
                    MERGE (v:Video {title: $video_title, url: $video_url})
                    MERGE (c)-[:MITIGATES]->(r)
                    MERGE (c)-[:MENTIONED_IN]->(v)
                    MERGE (r)-[:MENTIONED_IN]->(v)
                """, control=control, risk=risk, video_title=video_title, video_url=video_url)

def process_youtube_video(url):
    temp_dir = tempfile.mkdtemp()
    try:
        print("Downloading video...")
        video_file, video_title = download_youtube_video(url, temp_dir)
        print(f"Video title: {video_title}")
        print("Extracting audio...")
        audio_file = os.path.join(temp_dir, "audio.mp3")
        extract_audio(video_file, audio_file)
        print("Transcribing audio with timestamps...")
        segments = transcribe_audio_with_timestamps(audio_file)
        print(f"Transcribed {len(segments)} segments.")

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.new(embeddings)

        for i, segment in enumerate(segments):
            segment_text = segment['text']
            start_time = segment['start']
            end_time = segment['end']
            midpoint_time = (start_time + end_time) / 2
            image_file = os.path.join(temp_dir, f"frame_{i}.jpg")
            extract_screenshot(video_file, midpoint_time, image_file)
            image_text = extract_text_from_image(image_file)
            image_summary = summarize_image(image_file)
            combined_text = f"{segment_text}\nOCR Text from Image:\n{image_text}\nImage Summary:\n{image_summary}"

            # Include video title and URL in metadata
            doc = Document(
                page_content=combined_text,
                metadata={
                    'segment_index': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'video_title': video_title,
                    'video_url': url
                }
            )
            vector_store.add_documents([doc])

            # Extract security controls and risks using LLM agent
            security_info = extract_security_info(combined_text)

            # Store security info in Neo4j
            store_security_info_in_neo4j(driver, security_info, video_title, url)

        faiss_index_file = os.path.join(temp_dir, "faiss_index")
        vector_store.save_local(faiss_index_file)

        print("Processing complete.")

    finally:
        shutil.rmtree(temp_dir)

# Example usage
youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID"
process_youtube_video(youtube_url)