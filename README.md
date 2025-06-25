# Artificial Intelligence [Engineering] – UAB (2024–2025)

This repository contains solutions to two practical projects developed as part of the **Artificial Intelligence** course at **Universitat Autònoma de Barcelona** (course code: MO73906), academic year 2024–2025.

---

## 🗺️ Project 1 – Navigation System

**Goal**: Implement a navigation system to compute optimal metro routes in the city of Lyon, based on different search strategies and user preferences (e.g., shortest path, minimal cost).

### 🔍 Implemented Search Algorithms:
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- Uniform Cost Search (UCS)
- A* Search

### 🗂 Files:
- `SearchAlgorithm.py`: All algorithms were implemented in this single file as required.
- Data: Metro maps (`Lyon_smallCity`, `Lyon_bigCity`) and city structure files.

---

## 🧷 Project 2 – Image Labeling System

**Goal**: Create a system that can automatically classify clothing catalog images by **color** and **shape (type of garment)** using unsupervised and supervised learning methods.

### 🧠 Project Structure:
#### 📦 Part 1 – KMeans & Color
- Implemented KMeans clustering to detect the dominant colors in each image.
- File: `Kmeans.py`

#### 📦 Part 2 – KNN & Shape
- Used a K-Nearest Neighbors (KNN) classifier to identify garment type.
- File: `KNN.py`

#### 📦 Part 3 – Integration & Performance Analysis
- Combined both classifiers into a single labeling pipeline.
- Added performance metrics: accuracy, confusion matrix, F1-score, and runtime.
- File: `My_labeling.py`
- Final submission includes: `Kmeans.py`, `KNN.py`, `My_labeling.py`

---

## 📚 Technologies
- Python 3
- NumPy
- PIL / OpenCV (for image handling)
- Custom implementations (no external ML libraries used for KMeans/KNN)

---

## 📎 Notes
All algorithms were implemented from scratch, following the course requirements. The code is modular and documented to facilitate evaluation and reuse.

> 🏫 Course: Artificial Intelligence [Engineering]  
> 📍 Universitat Autònoma de Barcelona  
> 📅 Academic Year: 2024–2025

---
