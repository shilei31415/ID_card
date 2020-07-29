python savePB/optimize_for_inference.py  --input savePB/model.pb --output savePB/model_new.pb --frozen_graph True --input_names "image" --output_names "result"
python savePB/pb2pbtxt.py
