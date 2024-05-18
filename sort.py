import os
import hashlib
import json
import cv2
from transformers import pipeline
from pathlib import Path
from collections import Counter
import re
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import chardet
import magic
from tenacity import retry, stop_after_attempt, wait_exponential
import openai


# Configuration
CONFIG = {
    "api_key": "key",
    "max_threads": 10
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

openai.api_key = CONFIG["api_key"]

# Helper functions
def read_file_in_chunks(file_path, chunk_size=4096):
    """Generator to read a file in chunks."""
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Main functions
def list_files(startpath):
    file_details = {'text': [], 'image': [], 'video': [], 'other': []}
    for root, dirs, files in os.walk(startpath):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            mime_type = magic.from_file(file_path, mime=True)
            file_info = {'path': file_path, 'mime_type': mime_type}
            if 'text' in mime_type:
                file_details['text'].append(file_info)
            elif 'image' in mime_type:
                file_details['image'].append(file_info)
            elif 'video' in mime_type:
                file_details['video'].append(file_info)
            else:
                file_details['other'].append(file_info)
    return file_details

def find_duplicates(file_details):
    hashes = {}
    duplicates = {}
    for category, files in file_details.items():
        for file_info in tqdm(files, desc=f"Finding duplicates in {category}"):
            file_path = file_info['path']
            try:
                hasher = hashlib.sha256()
                for chunk in read_file_in_chunks(file_path):
                    hasher.update(chunk)
                file_hash = hasher.hexdigest()
                if file_hash in hashes:
                    duplicates.setdefault(file_hash, []).append(file_path)
                else:
                    hashes[file_hash] = file_path
            except (OSError, IOError) as e:
                logger.error(f"Error processing {file_path}: {e}")
    return duplicates

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(10)
    return [word for word, freq in common_words if freq > 1 and len(word) > 3]

def analyze_text(file_info):
    classifier = pipeline('sentiment-analysis')
    file_path = file_info['path']
    try:
        with open(file_path, 'rb') as file:
            detected_encoding = chardet.detect(file.read())['encoding']
        with open(file_path, 'r', encoding=detected_encoding) as file:
            text = file.read()
            sentiment_result = classifier(text[:1024])  # limit size for large texts
            keywords = extract_keywords(text)
            return {
                'sentiment': sentiment_result,
                'keywords': keywords
            }
    except (OSError, IOError, UnicodeDecodeError) as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_image(file_info):
    try:
        with open(file_info['path'], 'rb') as f:
            image_data = f.read()
        response = openai.Image.create(
            model="GPT-4-turbo",
            n=1,
            file=image_data
        )
        if 'data' in response:
            tags = [resp['text'] for resp in response['data']]
            return {'tags': tags}
        else:
            logger.warning(f"No data key in response for {file_info['path']}")
            return {'tags': []}
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error for {file_info['path']}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unhandled exception for {file_info['path']}: {str(e)}")
        return None

def extract_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    step = frame_count // num_frames
    for i in range(num_frames):
        frame_id = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            img_path = f'{os.path.splitext(video_path)[0]}_frame_{i}.jpg'
            cv2.imwrite(img_path, frame)
            frames.append(img_path)
    cap.release()
    for frame in frames:
        try:
            yield frame
        finally:
            os.remove(frame)

def analyze_video(file_info):
    video_path = file_info['path']
    try:
        frames = extract_frames(video_path)
        frame_results = {}
        with ThreadPoolExecutor(max_workers=CONFIG["max_threads"]) as executor:
            future_to_frame = {executor.submit(analyze_image, {'path': frame}): frame for frame in frames}
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                frame_result = future.result()
                if frame_result is not None:
                    frame_results[frame] = frame_result
        for frame in frames:
            os.remove(frame)
        return frame_results
    except (OSError, IOError, cv2.error) as e:
        logger.error(f"Error processing {video_path}: {e}")
        return None

def process_files(file_details):
    text_results = {}
    image_results = {}
    video_results = {}
    with ThreadPoolExecutor(max_workers=CONFIG["max_threads"]) as executor:
        text_futures = {executor.submit(analyze_text, file_info): file_info for file_info in file_details['text']}
        image_futures = {executor.submit(analyze_image, file_info): file_info for file_info in file_details['image']}
        video_futures = {executor.submit(analyze_video, file_info): file_info for file_info in file_details['video']}
        for future in as_completed(text_futures):
            file_info = text_futures[future]
            result = future.result()
            if result is not None:
                text_results[file_info['path']] = result
        for future in as_completed(image_futures):
            file_info = image_futures[future]
            result = future.result()
            if result is not None:
                image_results[file_info['path']] = result
        for future in as_completed(video_futures):
            file_info = video_futures[future]
            result = future.result()
            if result is not None:
                video_results.update(result)
    return text_results, image_results, video_results

def organize_files(analysis_results, base_path, classification_rules):
    for file_path, data in analysis_results.items():
        folder_name = "Uncategorized"  # Default if no rules match
        if 'keywords' in data:  # For text files
            for keyword in data['keywords']:
                if keyword in classification_rules['text']:
                    folder_name = classification_rules['text'][keyword]
                    break
        elif 'tags' in data:  # For images
            for tag in data['tags']:
                if tag in classification_rules['image']:
                    folder_name = classification_rules['image'][tag]
                    break
        target_dir = os.path.join(base_path, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        try:
            shutil.move(file_path, os.path.join(target_dir, os.path.basename(file_path)))
        except (OSError, IOError) as e:
            logger.error(f"Error moving {file_path}: {e}")

def load_classification_rules(path):
    with open(path, 'r') as file:
        return json.load(file)

def visualize_file_distribution(file_details):
    labels = []
    sizes = []
    for category, files in file_details.items():
        labels.append(category)
        # Ensure we only sum sizes of existing files
        total_size = sum(os.path.getsize(f['path']) for f in files if os.path.exists(f['path']))
        sizes.append(total_size)

    # Avoid plotting if all sizes are zero
    if all(size == 0 for size in sizes):
        logger.info("No files to visualize")
        return

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('File Distribution by Type and Size')
    plt.show()

def build_knowledge_graph(file_details, text_results, image_results, video_results):
    G = nx.Graph()
    for category in file_details:
        for file_info in file_details[category]:
            G.add_node(file_info['path'], type=category, mime=file_info['mime_type'])
    for path, data in text_results.items():
        keywords = data.get('keywords', [])
        for other_path, other_data in text_results.items():
            if path != other_path:
                other_keywords = other_data.get('keywords', [])
                shared_keywords = set(keywords) & set(other_keywords)
                if shared_keywords:
                    G.add_edge(path, other_path, weight=len(shared_keywords))
    for path, data in image_results.items():
        tags = data.get('tags', [])
        for other_path, other_data in image_results.items():
            if path != other_path:
                other_tags = other_data.get('tags', [])
                shared_tags = set(tags) & set(other_tags)
                if shared_tags:
                    G.add_edge(path, other_path, weight=len(shared_tags))
    return G

def plot_interactive_graph(G):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[], line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()

def main():
    directory_path = '../../../test-down'
    classification_rules = load_classification_rules('classification_rules.json')
    file_details = list_files(directory_path)
    duplicates = find_duplicates(file_details)
    logger.info(f"Duplicates Found: {duplicates}")
    text_results, image_results, video_results = process_files(file_details)
    organize_files(text_results, directory_path, classification_rules)
    organize_files(image_results, directory_path, classification_rules)
    organize_files(video_results, directory_path, classification_rules)
    G = build_knowledge_graph(file_details, text_results, image_results, video_results)
    plot_interactive_graph(G)
    visualize_file_distribution(file_details)

if __name__ == "__main__":
    main()
