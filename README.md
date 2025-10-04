# Probabilistic Routing Algorithm for Public Transport under Uncertainty

## Overview
This project implements a **stochastic routing algorithm** that accounts for real-world uncertainties in public transport systems, such as **delays**, **weather effects**, and **missed connections**.  
Given a desired arrival time and a confidence threshold (Q%), the algorithm computes one or more routes that guarantee arrival **before the given time with at least Q% probability**.

## Problem Statement
Traditional shortest-path algorithms (e.g., Dijkstra’s) assume deterministic travel times.  
However, real-world transport systems exhibit stochastic delays and transfer uncertainties.  
The goal of this project is to extend classical routing methods to model **uncertainty** and **compute probabilistic guarantees** on arrival times.

##  Methodology
1. **Graph Modeling**  
   - Represented the **Swiss public transport network** as a **weighted directed graph**, where nodes are stops and edges represent trips or transfers.  
   - Edge weights correspond to travel times with associated uncertainty distributions.

2. **Route Generation**  
   - Applied **Dijkstra’s algorithm** to find the baseline shortest path.  
   - Used **Yen’s algorithm** to compute *K* alternative paths.

3. **Confidence Estimation**  
   - For each edge (transfer), retrieved historical data to estimate delay distributions based on `(transfer type, weather)` pairs.  
   - Computed the **route-level confidence** by multiplying the probabilities of on-time arrival across all transfers.

4. **Route Ranking**  
   - Ranked the *K* candidate routes by their overall on-time arrival confidence.  
   - Selected the route(s) meeting or exceeding the target confidence threshold (≥ Q%).

## Data
- Historical delay data from Swiss public transport feeds.  
- Weather data from open meteorological APIs.  
- Combined and preprocessed to create empirical delay distributions.

## Key Features
- Probabilistic modeling of delays and transfer reliability.  
- Integration of Dijkstra’s and Yen’s algorithms for efficient multi-route computation.  
- Confidence-based ranking and filtering of routes.  
- Modular structure for easy adaptation to other transport networks.
