# Truth Table Solver & Verilog Generator

A Streamlit-based tool that solves Truth Tables using the Quine-McCluskey algorithm. It not only simplifies Boolean logic but also generates ready-to-use Verilog code and circuit diagrams.

## Features

- **Support for 2 to 5 Variables**: Handle complex logic problems with up to 5 inputs.
- **Interactive Truth Table**: Easily toggle outputs (`0`, `1`) directly in the UI.
- **K-Map Visualization**: View the Karnaugh Map for the given truth table.
- **Boolean Expression**: Get the simplified Sum-of-Products (SOP) expression in LaTeX format.
- **Verilog Generation**:
  - Automatically generates a structural/behavioral module for the logic.
  - Generates a corresponding testbench (`tb`) for verification.
  - Downloadable `.v` files.
- **Circuit Diagrams**: Visual representation of the logic circuit.
- **Waveform Simulation**: Visualizes the timing diagram for the logic inputs and output.

## Installation

This project is managed using `uv` for fast dependency management, but it also works with standard `pip`.

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (Recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Dash198/truth-table-solver.git
cd truth-table-solver

# Install dependencies using uv
uv venv
uv pip install -r requirements.txt
```

## Usage

You can run the application using the provided convenience script or directly via `uv`.

### Method 1: Helper Script (Recommended)
```bash
./run_app.sh
```

### Method 2: Manual Command
```bash
uv run streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the tool.

## How it Works

1. **Select Number of Variables**: Choose between 2, 3, 4, or 5 variables.
2. **Edit Truth Table**: Modify the "Y" (Output) column in the table.
3. **Solve**: Click the "Solve K-Map" button.
4. **Explore Results**:
   - **Expression Tab**: See the K-Map and the simplified equation.
   - **Verilog Tab**: View and download the Verilog module and testbench.
   - **Diagrams Tab**: View the circuit schematic and waveform.

## Implementation Details

- **Algorithm**: Uses the Quine-McCluskey method for finding prime implicants and essential prime implicants.
- **Visualization**: Uses `matplotlib` for waveforms and `schemdraw` for circuit diagrams.
- **GUI**: Built with `streamlit`.

## License

[MIT](LICENSE)
