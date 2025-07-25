# Trading Agent

This project implements various trading agents and trains them with synthetic courses. There are two different environments with slight differences in the observation function. For each agent, there is a training function and a test function. All courses used in the project are located in the 'synthetic_courses' folder. We implement a DQN, PPO, and A2C agent.

## Requirements

- Python
- pip

## Setup

It is **strongly recommended** to use a virtual environment:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

All agents are trained and tested in the 'testing_and_evaluating' folder. Each agent has a subfolder, which in turn contains two subfolders: 'evaluate' and 'training'. Under 'evaluate', you'll find a .py file that you can use to test the respective agent. There, you can enter the version of the agent you want to test as the model. Under the 'training' folder, you'll find a .py file that you can use to train an agent yourself. All you have to do is adjust the relevant attributes and change the name of the saved model. All models are saved in the 'models' folder within the respective subfolders.       The 'synthetic_courses' folder contains the training and test courses used, as well as plots representing the courses. There are also .py files that you can use to create new training and test courses.

## Project Structure

```bash
UNI-PROJEKT_RL/
├── logs/
├── models/
├── stock_exchange_env/
├── synthetic_courses/
├── testing_and_evaluating/
├── tests/
├── venv/
├── README.md
└── requirements.txt