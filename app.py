import streamlit as st
import pandas as pd
from kmap_solver import *
import itertools

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

if(st.button("Solve K-Map")):
    edited_df.columns = vars + ['Y']
    if(n_vars==2):
        
        kmap, expr, code, tb, img = solve_truth_table_2(edited_df)
        st.write("### K-map (Rows = B, Cols = A)")
        st.dataframe(kmap)
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
    
    elif n_vars==3:
        kmap, expr, code, tb, img = solve_truth_table_3(edited_df)
        st.write("### K-map (Rows = C, Cols = AB)")
        st.dataframe(kmap)
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
    
    elif n_vars==4:
        kmap, expr, code, tb, img = solve_truth_table_4(edited_df)
        st.write("### K-map (Rows = CD, Cols = AB)")
        st.dataframe(kmap)
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