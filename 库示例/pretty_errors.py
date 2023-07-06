#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:47:15 2022

@author: jack
"""

import pretty_errors

# 【重点】进行配置
pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    filename_color = pretty_errors.BRIGHT_YELLOW,
    line_number_first   = True,
    display_link        = True,
    lines_before        = 5,
    lines_after         = 2,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
    header_color        = 'blue',
    truncate_code       = 'True',
    display_locals      = 'True'
)

# 原来的代码
def foo():
    1/0

if __name__ == "__main__":
    foo()