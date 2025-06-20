import numpy as np 



def _gaussian_noise(dataset,std_dev):
    noised_dataset=[]
    for data in dataset:
        noise=np.random.normal(0,std_dev,len(data))
        noised_data=np.clip(noise+data,0,1)
        noised_dataset.append(noised_data)
    return np.array(noised_dataset)



_noise_functions={"gaussian":_gaussian_noise}

def get_noise_functions(function,std_deviation):
    def noise_func(dataset):
        return _noise_functions[function](dataset,std_deviation)
    return noise_func