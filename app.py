import streamlit as st
import pandas as pd
from kmap_solver import *
import itertools
import matplotlib.pyplot as plt
import numpy as np

st.title('Truth Table Solver')

n_vars = st.selectbox('Vars',options=[2,3,4])

st.write(f"You selected **{n_vars}** variables.")


# Step 2: generate variable names automatically
vars = [chr(ord('A')+i) for i in range(n_vars)]

# Step 3: generate all input combinations
combinations = list(itertools.product([0,1], repeat=n_vars))
df = pd.DataFrame(combinations, columns=vars)

# Add output column
df["Y"] = ""

# Step 4: let user edit the outputs
edited_df = st.data_editor(
    df,
    num_rows="fixed",
    column_config={
        "Y": st.column_config.SelectboxColumn("Y", options=["0","1","X"], required=True)
    }
)

col1, col2 = st.columns(2)

def plot_waveform(vars, truth_table):
    n = len(vars)
    rows = len(truth_table)

    fig, ax = plt.subplots(len(vars)+1, 1, figsize=(8, 2*(n+1)), sharex=True)

    # Generate timeline with an extra point for the final hold
    t = np.arange(rows+1)

    for i, var in enumerate(vars):
        y = truth_table[var].astype(int).to_numpy()
        y = np.append(y, y[-1])   # repeat last value
        ax[i].step(t, y, where='post')
        ax[i].set_ylabel(var)
        ax[i].set_yticks([0,1])
        ax[i].set_xlim(0, rows)  # make sure axis shows full length

    y = truth_table["Y"].astype(int).to_numpy()
    y = np.append(y, y[-1])       # repeat last value
    ax[-1].step(t, y, where='post', color='red')
    ax[-1].set_ylabel("Y")
    ax[-1].set_yticks([0,1])
    ax[-1].set_xlabel("Time")
    ax[-1].set_xlim(0, rows)

    plt.tight_layout()
    return fig


if(st.button("Solve K-Map")):
    edited_df.columns = vars + ['Y']
    if(n_vars==2):
        kmap, expr, code, tb, img = solve_truth_table_2(edited_df)

        latex_expr = expr.replace("&", r"\cdot") \
                        .replace("|", r"+") 
        
        tab1, tab2, tab3 = st.tabs(
        ["Expression", "Verilog", "Diagrams"]
        )

        with tab1:
            st.write("### K-map (Rows = B, Cols = A)")
            st.dataframe(kmap)

            st.write("### Expression")
            st.latex(f'Y = {latex_expr}')

        with tab2:
            st.write("### Verilog")
            st.code(code, language="verilog")
            st.download_button('Download Code (f.v)',code,file_name='f.v')
            st.write("### Testbench")
            st.code(tb, language="verilog")
            st.download_button('Download Code (f_tb.v)',tb,file_name='f_tb.v')

        with tab3:
            st.write("### Circuit Diagram")
            st.markdown(img, unsafe_allow_html=True)

            st.write("### Waveform")
            st.pyplot(plot_waveform(vars, edited_df))

    elif n_vars==3:
        kmap, expr, code, tb, img = solve_truth_table_3(edited_df)
        latex_expr = expr.replace("&", r"\cdot") \
                        .replace("|", r"+") 
        
        tab1, tab2, tab3 = st.tabs(
        ["Expression", "Verilog", "Diagrams"]
        )

        with tab1:
            st.write("### K-map (Rows = B, Cols = A)")
            st.dataframe(kmap)

            st.write("### Expression")
            st.latex(f'Y = {latex_expr}')

        with tab2:
            st.write("### Verilog")
            st.code(code, language="verilog")
            st.download_button('Download Code (f.v)',code,file_name='f.v')
            st.write("### Testbench")
            st.code(tb, language="verilog")
            st.download_button('Download Code (f_tb.v)',tb,file_name='f_tb.v')

        with tab3:
            st.write("### Circuit Diagram")
            st.markdown(img, unsafe_allow_html=True)

            st.write("### Waveform")
            st.pyplot(plot_waveform(vars, edited_df))
    
    elif n_vars==4:
        kmap, expr, code, tb, img = solve_truth_table_4(edited_df)
        latex_expr = expr.replace("&", r"\cdot") \
                        .replace("|", r"+") 
        
        tab1, tab2, tab3 = st.tabs(
        ["Expression", "Verilog", "Diagrams"]
        )

        with tab1:
            st.write("### K-map (Rows = B, Cols = A)")
            st.dataframe(kmap)

            st.write("### Expression")
            st.latex(f'Y = {latex_expr}')

        with tab2:
            st.write("### Verilog")
            st.code(code, language="verilog")
            st.download_button('Download Code (f.v)',code,file_name='f.v')
            st.write("### Testbench")
            st.code(tb, language="verilog")
            st.download_button('Download Code (f_tb.v)',tb,file_name='f_tb.v')

        with tab3:
            st.write("### Circuit Diagram")
            st.markdown(img, unsafe_allow_html=True)

            st.write("### Waveform")
            st.pyplot(plot_waveform(vars, edited_df))

    if n_vars == 5:
        kmap, expr, code, tb, img = solve_truth_table_5(edited_df)
        st.write("### K-map (Rows = CD, Cols = AB, E=0)")
        st.dataframe(kmap[0])
        st.write("### K-map (Rows = CD, Cols = AB, E=1)")
        st.dataframe(kmap[1])
        st.write("### Expression")
        # Convert your symbols to LaTeX
        latex_expr = expr.replace("&", r"\cdot") \
                        .replace("|", r"+") \

        st.latex(f'Y = {latex_expr}')

        st.write("### Verilog")
        st.code(code, language="verilog")
        st.write("### Testbench Code")
        st.code(tb, language="verilog")
        st.write("### Circuit Diagram")
        st.markdown(img, unsafe_allow_html=True)