> @Author: WANG Shixiong (Email: <s.wang@u.nus.edu>; <wsx.gugo@gmail.com>)

> @Affiliate: Department of Electrical & Electronic Engineering, Imperial College London

> @Date: 21 March 2024

> @Acknowledgement: https://mathworks.com/help/phased/ug/waveform-design-for-a-dual-function-mimo-radcom-system.html

> @MATLAB Dependencies: Phased Array System Toolbox, Communications Toolbox, Signal Processing Toolbox, DSP System Toolbox

> MATLAB Version: 2023a or later

# Robust-Waveform: Robust Waveform Design

Online supplementary materials of the paper titled 

**Robust Waveform Design for Integrated Sensing and Communication**

Published in the _IEEE Transactions on Signal Processing_ (DOI: [10.1109/TSP.2024.3410142](https://ieeexplore.ieee.org/document/10549948))

Authored by Shixiong Wang, Wei Dai, Haowei Wang, and Geoffrey Ye Li

Affiliated with the Department of Electrical & Electronic Engineering, Imperial College London, and Rice-Rick Digitalization, Singapore

**Codes**

- The file "main.m" is the entry.

- The main folder contains all core functions (i.e., the algorithmic implementations of Methods 1, 2, and 3; see Table I).

- The folder "Utils" includes supporting, but marginally related, functions.

- The folder "Externals" includes functions not written by me; they are from [here](https://mathworks.com/help/phased/ug/waveform-design-for-a-dual-function-mimo-radcom-system.html).

**Dependencies**

Please install the following MATLAB toolboxes before you run the codes:

```
Phased Array System Toolbox, Communications Toolbox, Signal Processing Toolbox, DSP System Toolbox
```


**Warrant**

Files/codes here are allowed to be edited, distributed, and re-used for any academic/teaching purpose without any warrant. However, you are strongly suggested to share your codes with the public if you are planning to use codes here. Let's work together to guarantee the reproducibility of experiments and the verifiability of claims in publications. We believe that this is meaningful to facilitate future research of the signal processing community.

**Disclaimer**

Note that the mentioned reproducibility and verifiability do not necessarily guarantee the (absolute) correctness of academic claims in a scientific publication. Future research may deny, modify, or improve the philosophies, methods, models, and/or claims conveyed in this article. But readers should not try to "find bones from an egg", and the codes here are just for their reference, not for their unfriendly criticism. Of course, the authors are open to learning, and friendly comments are always welcomed.
