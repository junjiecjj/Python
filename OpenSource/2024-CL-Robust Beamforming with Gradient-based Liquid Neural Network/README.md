# GLNN
This repository is the Python implementation of paper _"[Robust Beamforming with Gradient-based Liquid Neural Network](https://ieeexplore.ieee.org/document/10620247)"_.

## Blog
English version : NotImplemented.

Chinese version : [Zhihu](https://zhuanlan.zhihu.com/p/711109469).

## Files in this repo
`main.py`: The main function. Can be directly run to get the results.

`utils.py`: This file contains the util functions. It also contains definition of system params.

`net.py`: This file defines and declares the GLNN and the params.

`CSIdyn64.mat`: An example of dataset for trial run.

`requirements.txt`: The requirement of the recommended environment.
## Reference
Should you find this work beneficial, **kindly grant it a star**!

To keep abreast of our research, **please consider citing**:
```plain text
Xinquan Wang, Fenghao Zhu, Chongwen Huang, Ahmed Alhammadi, Faouzi Bader, Zhaoyang Zhang, Chau Yuen, Merouane Debbah, "Robust Beamforming with Gradient-based Liquid Neural Network," IEEE Wireless communications Letters.
```
```bibtex
@article{glnn,
      title={Robust Beamforming with Gradient-based Liquid Neural Network},
      author={Xinquan Wang and Fenghao Zhu and Chongwen Huang and Ahmed Alhammadi and Faouzi Bader and Zhaoyang Zhang and Chau Yuen and M{\'e}rouane Debbah},
      journal={IEEE Wirel. Commun. Lett.},
      year={2024}
}
```

Due to the size limitation of the files, bigger dataset is available [here](https://drive.google.com/file/d/1-luLm9BwtGcT-SoJt9IZUAEpsLTFPdpo/view?usp=drive_link).

## More than GLNN...
We are excited to announce a novel method that utilizes **Manifold Learning** to optimize spectrum efficiency in beamforming in RIS-aided MIMO systems. 

Compared to baseline, it can speed up the convergence by **23 times** and achieves a **stronger robustness**! 

See [GMML](https://github.com/FenghaoZhu/GMML/tree/main) for more information!
