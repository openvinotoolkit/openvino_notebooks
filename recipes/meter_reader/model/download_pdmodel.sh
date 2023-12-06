wget https://bj.bcebos.com/paddlex/examples2/meter_reader/meter_det_model.tar.gz
wget https://bj.bcebos.com/paddlex/examples2/meter_reader/meter_seg_model.tar.gz

mkdir analog

tar -xvf meter_det_model.tar.gz -C ./analog
tar -xvf meter_seg_model.tar.gz -C ./analog

rm meter_det_model.tar.gz meter_seg_model.tar.gz