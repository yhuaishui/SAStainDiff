SAMPLE_FLAGS="--input_path ./data/test/ --save_path ./data/sample/ --model_path ./checkpoint/other_to_Aperio.pt --use_ddim True --timestep_step 850 --timestep_respacing ddim5 --use_anysize False"

python test.py $SAMPLE_FLAGS

