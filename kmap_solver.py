import pandas as pd

def determine_state(var, value):
    if value=='1':
        return var
    else:
        return var+"\'"

def gen_state(var, value):
    state = ""
    for v, va in zip(var,value):
        state+=determine_state(v,va)+" & "

    return state[:-3]

def gen_boolean_map(vars, row_order, col_order):
    b_map = [["" for _ in range(len(col_order))] for _ in range(len(row_order))]
    for row_val in row_order:
        for col_val in col_order:
            b_map[row_order.index(row_val)][col_order.index(col_val)] = gen_state(vars,col_val+row_val)
    
    return pd.DataFrame(b_map)

def solve_kmap(kmap,bmap):
    m,n = kmap.shape
    expr = ""
    for i in range(m):
        for j in range(n):
            if(kmap.iloc[i,j]=="1"):
                expr += "(" + bmap.iloc[i,j] + ")" + " | "

    return expr[:-3]

def infix_to_postfix(expr):
    precedence = {'~':3, '&':2, '|':1}
    tokens = expr.replace('(', ' ( ').replace(')',' ) ').split()
    output, stack = [], []

    for t in tokens:
        if t.isalnum() or t.endswith("'"):
            if(t.endswith("'")):
                stack.append("~")
            output.append(t[0])
        
        elif t in precedence:
            while stack and stack[-1]!='(' and precedence[stack[-1]]>=precedence[t]:
                output.append(stack.pop())
            stack.append(t)
        
        elif t=='(':
            stack.append(t)
        elif t==')':
            while stack and stack[-1]!='(':
                output.append(stack.pop())
            stack.pop()
    
    while stack:
        output.append(stack.pop())
    return output

def postfix_to_verilog(postfix_stack, vars=['A','B']):
    stack = []
    for token in postfix_stack:
        if token.isalnum():
            stack.append(token)
        else:
            if(token=="~"):
                stack.append(token+stack.pop())
            else:
                b = stack.pop()
                stack.append("(" + stack.pop() + " " + token + " " + b + ")")
    
    inputs = ", ".join(f"input {x}" for x in vars)
    y = stack.pop()
    out = f"""module f({inputs}, output Y);
    assign Y = {y};
endmodule"""
    return out

def map2(gray_order,a,b):
    return gray_order.index(str(a)+str(b))

def gen_2_kmap(TRUTH_TABLE):
    kmap = [[-1 for _ in range(2)] for _ in range(2)]

    for a in range(2):
        for b in range(2):
            kmap[b][a] = TRUTH_TABLE[(TRUTH_TABLE['A']==a) & (TRUTH_TABLE['B']==b)].iloc[0,-1]
    df = pd.DataFrame(kmap)
    df.index = ['0','1']
    df.columns = ['0','1']
    return df

def gen_3_kmap(TRUTH_TABLE):
    kmap = [[0 for _ in range(4)] for _ in range(2)]
    gray_code_order = ['00','01','11','10']
    for a in range(2):
        for b in range(2):
            for c in range(2):
                kmap[c][map2(gray_code_order,a,b)] = TRUTH_TABLE[(TRUTH_TABLE['A']==a) & (TRUTH_TABLE['B']==b) & (TRUTH_TABLE['C']==c)].iloc[0,-1]

    kmap_df = pd.DataFrame(kmap,index=['0','1'])
    kmap_df.columns = gray_code_order
    return kmap_df

def gen_4_kmap(TRUTH_TABLE):
    kmap = [[0 for _ in range(4)] for _ in range(4)]
    gray_code_order = ['00','01','11','10']
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    kmap[map2(gray_code_order,c,d)][map2(gray_code_order,a,b)] = TRUTH_TABLE[(TRUTH_TABLE['A']==a) & (TRUTH_TABLE['B']==b) & (TRUTH_TABLE['C']==c) & (TRUTH_TABLE['D']==d)].iloc[0,-1]

    kmap_df = pd.DataFrame(kmap,index=gray_code_order)
    kmap_df.columns = gray_code_order
    return kmap_df

def gen_tb_2():
    out = """`timescale 1ns/1ps
module tb;
    reg A, B;
    wire Y;
    
    f uut (
        .A(A),
        .B(B),
        .Y(Y)
    );
    
    integer i;
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0,tb);
        
        for(i = 0; i < 4; i = i + 1) begin
            {A, B} = i;
            #5
        end
    
        $finish
    end
endmodule"""

    return out

def gen_tb_3():
    out = """`timescale 1ns/1ps
module tb;
    reg A, B, C;
    wire Y;
    
    f uut (
        .A(A),
        .B(B),
        .C(C),
        .Y(Y)
    );
    
    integer i;
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0,tb);
        
        for(i = 0; i < 8; i = i + 1) begin
            {A, B, C} = i;
            #5
        end
    
        $finish
    end
endmodule"""

    return out

def gen_tb_4():
    out = """`timescale 1ns/1ps
module tb;
    reg A, B, C, D;
    wire Y;
    
    f uut (
        .A(A),
        .B(B),
        .C(C),
        .D(D),
        .Y(Y)
    );
    
    integer i;
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0,tb);
        
        for(i = 0; i < 16; i = i + 1) begin
            {A, B, C, D} = i;
            #5
        end
    
        $finish
    end
endmodule"""

    return out

def gen_tb_5():
    out = """`timescale 1ns/1ps
module tb;
    reg A, B, C, D, E;
    wire Y;
    
    f uut (
        .A(A),
        .B(B),
        .C(C),
        .D(D),
        .E(E),
        .Y(Y)
    );
    
    integer i;
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0,tb);
        
        for(i = 0; i < 32; i = i + 1) begin
            {A, B, C, D, E} = i;
            #5
        end
    
        $finish
    end
endmodule"""

    return out

def solve_truth_table_2(TRUTH_TABLE):
    kmap = gen_2_kmap(TRUTH_TABLE)
    bmap = gen_boolean_map("AB",['0','1'],['0','1'])
    expr = solve_kmap(kmap,bmap)
    code = postfix_to_verilog(infix_to_postfix(expr))
    tb = gen_tb_2()

    return kmap, expr, code, tb

def solve_truth_table_3(TRUTH_TABLE):
    kmap = gen_3_kmap(TRUTH_TABLE)
    bmap = gen_boolean_map("ABC",['0','1'],['00','01','11','10'])
    expr = solve_kmap(kmap,bmap)
    code = postfix_to_verilog(infix_to_postfix(expr),vars=['A','B','C'])
    tb = gen_tb_3()

    return kmap,expr,code, tb

def solve_truth_table_4(TRUTH_TABLE):
    kmap = gen_4_kmap(TRUTH_TABLE)
    bmap = gen_boolean_map('ABCD',['00','01','11','10'],['00','01','11','10'])
    expr = solve_kmap(kmap,bmap)
    code = postfix_to_verilog(infix_to_postfix(expr),vars='ABCD')
    tb = gen_tb_4()

    return kmap, expr, code, tb

def solve_truth_table_5(TRUTH_TABLE):
    t0 = TRUTH_TABLE[TRUTH_TABLE['E']=='0'].reset_index(drop=True)
    t1 = TRUTH_TABLE[TRUTH_TABLE['E']=='1'].reset_index(drop=True)

    k0,e0,_ = solve_truth_table_4(t0)
    k1,e1,_ = solve_truth_table_4(t1)

    expr = f"(E' & {e0}) | (E & {e1})"
    code = postfix_to_verilog(infix_to_postfix(expr), vars=list("ABCDE"))
    tb = gen_tb_5()

    return (k0,k1), expr, code, tb