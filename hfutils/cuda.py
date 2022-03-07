COMPUTE_CAPACITY_CONFIG = {
'2.0': {
    'version': '2.0',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 48,
    'threadsPerMultiprocessor': 1536,
    'threadBlocksPerMultiprocessor': 8,
    'sharedMemoryPerMultiprocessor': 49152,
    'registerFileSize': 32768,
    'registerAllocationUnitSize': 64,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 63,
    'maxRegistersPerBlock': 32768,
    'sharedMemoryAllocationUnitSize': 128,
    'warpAllocationGranularity': 2,
    'maxThreadBlockSize': 1024
},
'2.1': {
    'version': '2.1',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 48,
    'threadsPerMultiprocessor': 1536,
    'threadBlocksPerMultiprocessor': 8,
    'sharedMemoryPerMultiprocessor': 49152,
    'registerFileSize': 32768,
    'registerAllocationUnitSize': 64,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 63,
    'maxRegistersPerBlock': 32768,
    'sharedMemoryAllocationUnitSize': 128,
    'warpAllocationGranularity': 2,
    'maxThreadBlockSize': 1024
},
'3.0': {
    'version': '3.0',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 16,
    'sharedMemoryPerMultiprocessor': 49152,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 63,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'3.2': {
    'version': '3.2',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 16,
    'sharedMemoryPerMultiprocessor': 49152,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'3.5': {
    'version': '3.5',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 16,
    'sharedMemoryPerMultiprocessor': 49152,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'3.7': {
    'version': '3.7',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 16,
    'sharedMemoryPerMultiprocessor': 114688,
    'registerFileSize': 131072,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'5.0': {
    'version': '5.0',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 32,
    'sharedMemoryPerMultiprocessor': 65536,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'5.2': {
    'version': '5.2',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 32,
    'sharedMemoryPerMultiprocessor': 98304,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 32768,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'5.3': {
    'version': '5.3',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 32,
    'sharedMemoryPerMultiprocessor': 65536,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 32768,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'6.0': {
    'version': '6.0',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 32,
    'sharedMemoryPerMultiprocessor': 65536,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 2,
    'maxThreadBlockSize': 1024
},
'6.1': {
    'version': '6.1',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 32,
    'sharedMemoryPerMultiprocessor': 98304,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'6.2': {
    'version': '6.2',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 32,
    'sharedMemoryPerMultiprocessor': 65536,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'7.0': {
    'version': '7.0',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 32,
    'sharedMemoryPerMultiprocessor': 98304,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'7.5': {
    'version': '7.5',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 32,
    'threadsPerMultiprocessor': 1024,
    'threadBlocksPerMultiprocessor': 16,
    'sharedMemoryPerMultiprocessor': 65536,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 256,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'8.0': {
    'version': '8.0',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 64,
    'threadsPerMultiprocessor': 2048,
    'threadBlocksPerMultiprocessor': 32,
    'sharedMemoryPerMultiprocessor': 167936,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 128,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
},
'8.6': {
    'version': '8.6',
    'threadsPerWarp': 32,
    'warpsPerMultiprocessor': 48,
    'threadsPerMultiprocessor': 1536,
    'threadBlocksPerMultiprocessor': 16,
    'sharedMemoryPerMultiprocessor': 102400,
    'registerFileSize': 65536,
    'registerAllocationUnitSize': 256,
    'allocationGranularity': 'warp',
    'maxRegistersPerThread': 255,
    'maxRegistersPerBlock': 65536,
    'sharedMemoryAllocationUnitSize': 128,
    'warpAllocationGranularity': 4,
    'maxThreadBlockSize': 1024
}
}

cudaRuntimeUsedSharedMemory = {
    '11.0': 1024,
    '11.1': 1024
}

import numpy as np

class CudaOccupancyCalculator:
    def __init__(self, compute_capacity) -> None:
        pass

    def set_inputs(self, threadsPerBlock: int, registersPerThread: int, compute_capacity: int, sharedMemoryPerBlock: int):
        self.reg_thd = registersPerThread
        self.config = COMPUTE_CAPACITY_CONFIG[compute_capacity]
        self.thd_blk = threadsPerBlock
        self.shm_blk = sharedMemoryPerBlock

    def blockWarps(self):
        threadsPerWarp = self.config['threadsPerWarp']
        return np.ceil(self.thd_blk / threadsPerWarp)

    def registersPerWarp(self):
        x = self.reg_thd * self.config['threadsPerWarp']
        registerAllocationUnitSize = self.config['registerAllocationUnitSize']
        return (x + registerAllocationUnitSize - 1) & (-registerAllocationUnitSize)

    def blockRegisters(self):
        return self.registersPerWarp() * self.blockWarps()

    def warpsPerMultiprocessorLimitedByRegisters(self):
        warpAllocationGranularity = self.config['warpAllocationGranularity']
        maxRegistersPerBlock = self.config['maxRegistersPerBlock']
        return int(maxRegistersPerBlock / self.registersPerWarp()) & (-warpAllocationGranularity)

    # def blockCudaRuntimeSharedMemory(compute_capacity, cuda_version=None):
    #     if float(compute_capacity) >= 8:
    #         return cudaRuntimeUsedSharedMemory[cuda_version]
    #     return 0

    def blockSharedMemory(self):
        sharedMemoryAllocationUnitSize = self.config['sharedMemoryAllocationUnitSize']
        return (self.shm_blk + sharedMemoryAllocationUnitSize - 1) & (-sharedMemoryAllocationUnitSize)

    def threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor(self):
        threadBlocksPerMultiprocessor = self.config['threadBlocksPerMultiprocessor']
        warpsPerMultiprocessor = self.config['warpsPerMultiprocessor']
        return min(
            threadBlocksPerMultiprocessor, 
            np.floor(warpsPerMultiprocessor / self.blockWarps())
        )

    def threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor(self):
        maxRegistersPerThread = self.config['maxRegistersPerThread']
        if self.reg_thd > maxRegistersPerThread:
            return 0
        elif self.reg_thd > 0:
            registerFileSize = self.config['registerFileSize']
            maxRegistersPerBlock = self.config['maxRegistersPerBlock']
            return np.floor(self.warpsPerMultiprocessorLimitedByRegisters() / self.blockWarps()) * np.floor(registerFileSize / maxRegistersPerBlock)
        
        threadBlocksPerMultiprocessor = self.config['threadBlocksPerMultiprocessor']
        return threadBlocksPerMultiprocessor

    def threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor(self):
        sharedMemoryPerMultiprocessor = self.config['sharedMemoryPerMultiprocessor']
        if self.shm_blk > 0:
            return np.floor(sharedMemoryPerMultiprocessor / self.blockSharedMemory())
        
        threadBlocksPerMultiprocessor = self.config['threadBlocksPerMultiprocessor']
        return threadBlocksPerMultiprocessor

    def activeThreadBlocksPerMultiprocessor(self):
        return min(
            self.threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor(), 
            self.threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor(), 
            self.threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor()
        )

    def activeThreadsPerMultiprocessor(self):
        return self.thd_blk * self.activeThreadBlocksPerMultiprocessor()
    
    def activeWarpsPerMultiprocessor(self):
        return self.activeThreadBlocksPerMultiprocessor() * self.blockWarps()
    
    def occupancyOfMultiprocessor(self):
        warpsPerMultiprocessor = self.config['warpsPerMultiprocessor']
        return self.activeWarpsPerMultiprocessor() / warpsPerMultiprocessor
