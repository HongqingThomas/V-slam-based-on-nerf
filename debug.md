# debugging
1. set src/Tracker.py's line64 dataloader num_worker = 0, release cpu resource 
2. add some print in utils/Visualizer.py
3. comment visualizer.py function in Mapper: line 88 and line 428
4. make apartment verbose = False
5. change tracker iteration in each epoch from 50 to 30
6. add     
    import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
to run.py
7. Tracker.py line 65 DataLoader()add pin_memory = True
8. add grid_len ro config/apartment
9. tracker.py 248 add torch.cuda.empty_cache()
10. render.py. mesher.py points_batch_size=500000, ray_batch_size=100000 -> points_batch_size=50000, ray_batch_size=10000
11. Mapper.py comment self.mesher.get_mesh (~)


# Error:
    RuntimeError: CUDA out of memory. Tried to allocate 116.00 MiB (GPU 0; 23.68 GiB total capacity; 1.03 GiB already allocated; 16.66 GiB free; 1.10 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF


# TODO:
1. bound readme.md
2. Mesher.py line 385 # points does not need to put into cuda

1. 引进dataset
2. config/highbay
3. src/dataset.py class highbay