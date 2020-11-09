from main import model_attack

"""
model_names can be the sublist of ['irse101','ir34','ir50','ir100','ir152']
attack_methods can be the combination of the "ti","mi","si","di"

input_path: The path of the original images' folder
output_path: The path of the attacked images' folder
pair_path: The path of the pair.txt
ts: shreshold
dataset: can be 'test' or 'val'
max_step: The maximum attack steps
alpha: Attack step size for each step
weights: The weights for every model. If weights is None,the all models' are the same value. 
"""

model_names = ['irse101','ir34','ir50','ir100','ir152']
attack_methods = 'sidi'
model = model_attack(model_names,attack_methods)

input_path = r'./images/input/test'
output_path = r'./images/output/test'
pair_path = r'./images/input/test/pair.txt'

ts = 0.9
dataset = 'test'
max_step = 300
alpha = 0.5
# weights = [1/5] * 5
model(input_path,output_path,pair_path,ts,dataset,max_step,alpha,weights = None)