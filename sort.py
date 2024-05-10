import os
import mimetypes
import shutil
import hashlib
import json
import requests
import cv2
from transformers import pipeline
from pathlib import Path
from collections import Counter
import re
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

# Function to list files and categorize by type
def list_files(startpath):
    file_details = {'text': [], 'image': [], 'video': [], 'other': []}
    for root, dirs, files in os.walk(startpath):
        for f in files:
            file_path = os.path.join(root, f)
            type, encoding = mimetypes.guess_type(file_path)
            mod_time = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
            file_info = {'path': file_path, 'mod_time': mod_time}
            if type:
                if 'text' in type:
                    file_details['text'].append(file_info)
                elif 'image' in type:
                    file_details['image'].append(file_info)
                elif 'video' in type:
                    file_details['video'].append(file_info)
                else:
                    file_details['other'].append(file_info)
            else:
                file_details['other'].append(file_info)
    return file_details

# Function to find duplicates based on file hash
def find_duplicates(file_paths):
    hashes = {}
    duplicates = {}
    for file_info in file_paths:
        file_path = file_info['path']
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash in hashes:
                    if file_hash in duplicates:
                        duplicates[file_hash].append(file_path)
                    else:
                        duplicates[file_hash] = [hashes[file_hash], file_path]
                else:
                    hashes[file_hash] = file_path
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return duplicates

# Function to extract keywords from text
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(10)
    return [word for word, freq in common_words if freq > 1 and len(word) > 3]

# Function to analyze text files
def analyze_text(file_details):
    classifier = pipeline('sentiment-analysis')
    results = {}
    for file_info in file_details:
        file_path = file_info['path']
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            try:
                sentiment_result = classifier(text[:1024])  # limit size for large texts
                keywords = extract_keywords(text)
                results[file_path] = {
                    'sentiment': sentiment_result,
                    'keywords': keywords
                }
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return results

# Function to analyze images using OpenAI API
def analyze_image_with_openai(file_details):
    headers = {
        'Authorization': 'Bearer YOUR_OPENAI_API_KEY',
        'Content-Type': 'application/json',
    }
    results = {}
    for file_info in file_details:
        file_path = file_info['path']
        with open(file_path, 'rb') as f:
            response = requests.post(
                'https://api.openai.com/v1/images/generations',
                headers=headers,
                json={'model': 'QPT-5-turbo', 'n': 1, 'image': f.read()}
            )
            response_data = response.json()
            tags = [resp['text'] for resp in response_data['data']] if response_data.get('data') else []
            results[file_path] = {'tags': tags}
    return results

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
            img_path = f'frame_{i}.jpg'
            cv2.imwrite(img_path, frame)
            frames.append(img_path)
    cap.release()
    return frames

def analyze_video_files(file_details):
    results = {}
    for file_info in file_details:
        video_path = file_info['path']
        frames = extract_frames(video_path)
        frame_results = analyze_image_with_openai([{'path': frame} for frame in frames])
        results[video_path] = frame_results
    return results

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
        shutil.move(file_path, os.path.join(target_dir, os.path.basename(file_path)))

# Function to load classification rules from a JSON file
def load_classification_rules(path):
    with open(path, 'r') as file:
        return json.load(file)

# Function to visualize file distribution
def visualize_file_distribution(file_details):
    labels = []
    sizes = []
    for category, files in file_details.items():
        labels.append(category)
        total_size = sum(os.path.getsize(f['path']) for f in files)
        sizes.append(total_size)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('File Distribution by Type and Size')
    plt.show()

# Function to plot an interactive knowledge graph
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

def build_knowledge_graph(file_details, text_results, image_results, video_results):
    G = nx.Graph()

    # Add nodes for each file
    for category in file_details:
        for file_info in file_details[category]:
            G.add_node(file_info['path'])

    # Define a simple relationship: files in the same category share an edge
    for category, results in [('text', text_results), ('image', image_results), ('video', video_results)]:
        for path, data in results.items():
            keywords = data.get('keywords', []) + data.get('tags', [])
            for other_path, other_data in results.items():
                if path != other_path:
                    other_keywords = other_data.get('keywords', []) + other_data.get('tags', [])
                    # Check if they share any keyword or tag
                    if any(kw in other_keywords for kw in keywords):
                        G.add_edge(path, other_path)
    return G


if __name__ == "__main__":
    directory_path = 'your_directory_path'
    classification_rules = load_classification_rules('classification_rules.json')
    file_details = list_files(directory_path)
    duplicates = find_duplicates([f['path'] for sublist in file_details.values() for f in sublist])
    print("Duplicates Found:", duplicates)
    text_results = analyze_text(file_details['text'])
    image_results = analyze_image_with_openai(file_details['image'])
    video_results = analyze_video_files(file_details['video'])
    organize_files(text_results, directory_path, classification_rules)
    organize_files(image_results, directory_path, classification_rules)
    organize_files(video_results, directory_path, classification_rules)
    G = build_knowledge_graph(file_details, text_results, image_results, video_results)
    plot_interactive_graph(G)
    visualize_file_distribution(file_details)

