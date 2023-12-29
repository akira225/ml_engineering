import os
from pathlib import Path
cwd = os.getcwd()


rule predict:
    input:
        model = os.path.join(cwd,Path('models/models.bin')),
        test = os.path.join(cwd, Path('data/preprocessed/test.csv'))
    output:
        predict=os.path.join(cwd, Path('data/predictions/predict.csv'))
    params:
        file=os.path.join(cwd,Path('src/predict.py'))
    shell:
        '''python "{params.file}" --data "{input.test}" --model "{input.model}" --out "{output.predict}"'''


rule train_model:
    input:
        train=os.path.join(cwd,Path('data/preprocessed/train.csv')),
        val=os.path.join(cwd,Path('data/preprocessed/val.csv'))
    output:
        model_dir = os.path.join(cwd,Path('models/models.bin')),
        report_dir = os.path.join(cwd,Path('reports/reports.txt'))
    params:
        file = os.path.join(cwd, Path('src/train.py'))
    shell:
        '''python "{params.file}" --train "{input.train}" --val "{input.val}" --model_dir "{output.model_dir}" --report_dir "{output.report_dir}"'''


rule prepare_data:
    input:
        data = os.path.join(cwd, Path('data/raw/train.csv')),
        data_test = os.path.join(cwd, Path('data/raw/test.csv'))
    output:
        train = os.path.join(cwd, Path('data/preprocessed/train.csv')),
        val = os.path.join(cwd, Path('data/preprocessed/val.csv')),
        test = os.path.join(cwd, Path('data/preprocessed/test.csv')),
    params:
        file = os.path.join(cwd, Path('src/prepare_data.py'))
    shell:
        '''python "{params.file}"  --data "{input.data}" --test_data "{input.data_test}" --train "{output.train}" --val "{output.val}" --test "{output.test}"'''



