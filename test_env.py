#!/usr/bin/env python3
"""
Minimal Test App for Pandas Alignment

This app creates two simple pandas Series with the same index,
multiplies them using the pandas .mul() method (with fill_value=0),
and displays the original Series, their shapes, and the result.
"""

import streamlit as st
import pandas as pd

def main():
    st.title("Minimal Test for Pandas Alignment")
    
    # Create two Series with identical indices
    s1 = pd.Series([10, 20, 30], index=[0, 1, 2])
    s2 = pd.Series([1, 2, 3], index=[0, 1, 2])
    
    # Display the Series
    st.subheader("Series s1:")
    st.write(s1)
    st.subheader("Series s2:")
    st.write(s2)
    
    # Multiply the two Series using .mul() with fill_value=0
    result = s1.mul(s2, fill_value=0)
    
    # Display the result and shapes
    st.subheader("Result of Multiplication (s1 * s2):")
    st.write(result)
    st.subheader("Shapes of the Series:")
    st.write("s1 shape:", s1.shape)
    st.write("s2 shape:", s2.shape)

if __name__ == "__main__":
    main()
