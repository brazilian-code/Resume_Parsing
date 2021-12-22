#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
def labels(ax):
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x = p.get_x()
        y = p.get_y()
        ax.annotate(f"{int(height)}", (x + width/2, y + height*1.01), ha="center")
        plt.ylabel("Count of Resumes")




