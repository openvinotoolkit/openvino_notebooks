<?xml version="1.0" ?>
<net name="age_gender" version="11">
	<layers>
		<layer id="0" name="data" type="Parameter" version="opset1">
			<data shape="1,3,62,62" element_type="f32"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="data"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32" names="data">
					<dim>1</dim>
					<dim>3</dim>
					<dim>62</dim>
					<dim>62</dim>
					<rt_info>
						<attribute name="layout" version="0" layout="[N,C,H,W]"/>
					</rt_info>
				</port>
			</output>
		</layer>
		<layer id="1" name="data/scale_copy" type="Const" version="opset1">
			<data element_type="f32" shape="48, 3, 3, 3" offset="0" size="5184"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="data/scale_copy"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="conv1/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv1/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>62</dim>
					<dim>62</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="data_add_1242" type="Const" version="opset1">
			<data element_type="f32" shape="1, 48, 1, 1" offset="5184" size="192"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="data_add_1242"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="conv1/Fused_Add_" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv1/Fused_Add_"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="conv1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="pool1" type="MaxPool" version="opset8">
			<data strides="2, 2" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" kernel="3, 3" rounding_type="ceil" auto_pad="explicit" index_element_type="i64" axis="0"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="pool1"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>60</dim>
					<dim>60</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="pool1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="relu1" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu1"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="pool1">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="bn_conv2/mean/Fused_Mul__copy" type="Const" version="opset1">
			<data element_type="f32" shape="64, 48, 3, 3" offset="5376" size="110592"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="bn_conv2/mean/Fused_Mul__copy"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>64</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="conv2/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv2/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>48</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>48</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="data_add_12451250" type="Const" version="opset1">
			<data element_type="f32" shape="1, 64, 1, 1" offset="115968" size="256"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="data_add_12451250"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="conv2/Fused_Add_" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv2/Fused_Add_"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="conv2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="pool2" type="MaxPool" version="opset8">
			<data strides="2, 2" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" kernel="3, 3" rounding_type="ceil" auto_pad="explicit" index_element_type="i64" axis="0"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="pool2"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="pool2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="relu2" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu2"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="pool2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="bn_conv3/mean/Fused_Mul__copy" type="Const" version="opset1">
			<data element_type="f32" shape="96, 64, 3, 3" offset="116224" size="221184"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="bn_conv3/mean/Fused_Mul__copy"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>96</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="conv3/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="1, 1" pads_end="1, 1" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv3/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>96</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="15" name="data_add_12531258" type="Const" version="opset1">
			<data element_type="f32" shape="1, 96, 1, 1" offset="337408" size="384"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="data_add_12531258"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="conv3/Fused_Add_" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv3/Fused_Add_"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="conv3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="pool3" type="MaxPool" version="opset8">
			<data strides="2, 2" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" kernel="3, 3" rounding_type="ceil" auto_pad="explicit" index_element_type="i64" axis="0"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="pool3"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="pool3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
				<port id="2" precision="I64">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="relu3" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu3"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="pool3">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="19" name="bn_conv4/mean/Fused_Mul__copy" type="Const" version="opset1">
			<data element_type="f32" shape="192, 96, 3, 3" offset="337792" size="663552"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="bn_conv4/mean/Fused_Mul__copy"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>192</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="20" name="conv4/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv4/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>96</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>192</dim>
					<dim>96</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="21" name="data_add_12611266" type="Const" version="opset1">
			<data element_type="f32" shape="1, 192, 1, 1" offset="1001344" size="768"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="data_add_12611266"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="22" name="conv4/Fused_Add_" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv4/Fused_Add_"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="conv4">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="23" name="relu4" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu4"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="conv4">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="64" type="Const" version="opset1">
			<data element_type="f32" shape="256, 192, 3, 3" offset="1002112" size="1769472"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="64"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="25" name="conv5/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv5/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>192</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>192</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="conv5/Dims716" type="Const" version="opset1">
			<data element_type="f32" shape="1, 256, 1, 1" offset="2771584" size="1024"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv5/Dims716"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="27" name="conv5" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="conv5"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="conv5">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="28" name="relu5" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu5"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="conv5">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="44" type="Const" version="opset1">
			<data element_type="f32" shape="256, 256, 3, 3" offset="2772608" size="2359296"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="44"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="30" name="age_conv1/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv1/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="31" name="age_conv1/Dims722" type="Const" version="opset1">
			<data element_type="f32" shape="1, 256, 1, 1" offset="5131904" size="1024"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv1/Dims722"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="age_conv1" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv1"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="age_conv1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="33" name="relu6_a" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu6_a"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="age_conv1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="54" type="Const" version="opset1">
			<data element_type="f32" shape="512, 256, 1, 1" offset="5132928" size="524288"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="54"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="35" name="age_conv2/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv2/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="age_conv2/Dims752" type="Const" version="opset1">
			<data element_type="f32" shape="1, 512, 1, 1" offset="5657216" size="2048"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv2/Dims752"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="age_conv2" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv2"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="age_conv2">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="38" name="relu7_a" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu7_a"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="age_conv2">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="39" name="42" type="Const" version="opset1">
			<data element_type="f32" shape="1, 512, 1, 1" offset="5659264" size="2048"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="42"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="40" name="age_conv3/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv3/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="age_conv3/Dims746" type="Const" version="opset1">
			<data element_type="f32" shape="1, 1, 1, 1" offset="5661312" size="4"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv3/Dims746"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="age_conv3" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="age_conv3"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="age_conv3,fc3_a">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="44" name="74" type="Const" version="opset1">
			<data element_type="f32" shape="256, 256, 3, 3" offset="5661316" size="2359296"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="74"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="45" name="gender_conv1/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv1/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="46" name="gender_conv1/Dims704" type="Const" version="opset1">
			<data element_type="f32" shape="1, 256, 1, 1" offset="8020612" size="1024"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv1/Dims704"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="47" name="gender_conv1" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv1"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="gender_conv1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="relu6_g" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu6_g"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="gender_conv1">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="72" type="Const" version="opset1">
			<data element_type="f32" shape="512, 256, 1, 1" offset="8021636" size="524288"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="72"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="gender_conv2/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv2/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>512</dim>
					<dim>256</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="51" name="gender_conv2/Dims740" type="Const" version="opset1">
			<data element_type="f32" shape="1, 512, 1, 1" offset="8545924" size="2048"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv2/Dims740"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="52" name="gender_conv2" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv2"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="gender_conv2">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="53" name="relu7_g" type="ReLU" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="relu7_g"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="gender_conv2">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="54" name="56" type="Const" version="opset1">
			<data element_type="f32" shape="2, 512, 1, 1" offset="8547972" size="4096"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="56"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="55" name="gender_conv3/WithoutBiases" type="Convolution" version="opset1">
			<data strides="1, 1" dilations="1, 1" pads_begin="0, 0" pads_end="0, 0" auto_pad="explicit"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv3/WithoutBiases"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>2</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="56" name="gender_conv3/Dims710" type="Const" version="opset1">
			<data element_type="f32" shape="1, 2, 1, 1" offset="8552068" size="8"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv3/Dims710"/>
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="57" name="gender_conv3" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="gender_conv3"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="gender_conv3">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="58" name="prob" type="SoftMax" version="opset8">
			<data axis="1"/>
			<rt_info>
				<attribute name="fused_names" version="0" value="prob"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="prob">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="59" name="prob/sink_port_0" type="Result" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="prob/sink_port_0"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
		<layer id="43" name="fc3_a" type="Result" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="fc3_a"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="1" to-layer="8" to-port="0"/>
		<edge from-layer="7" from-port="0" to-layer="8" to-port="1"/>
		<edge from-layer="8" from-port="2" to-layer="10" to-port="0"/>
		<edge from-layer="9" from-port="0" to-layer="10" to-port="1"/>
		<edge from-layer="10" from-port="2" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="1" to-layer="12" to-port="0"/>
		<edge from-layer="12" from-port="1" to-layer="14" to-port="0"/>
		<edge from-layer="13" from-port="0" to-layer="14" to-port="1"/>
		<edge from-layer="14" from-port="2" to-layer="16" to-port="0"/>
		<edge from-layer="15" from-port="0" to-layer="16" to-port="1"/>
		<edge from-layer="16" from-port="2" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="18" to-port="0"/>
		<edge from-layer="18" from-port="1" to-layer="20" to-port="0"/>
		<edge from-layer="19" from-port="0" to-layer="20" to-port="1"/>
		<edge from-layer="20" from-port="2" to-layer="22" to-port="0"/>
		<edge from-layer="21" from-port="0" to-layer="22" to-port="1"/>
		<edge from-layer="22" from-port="2" to-layer="23" to-port="0"/>
		<edge from-layer="23" from-port="1" to-layer="25" to-port="0"/>
		<edge from-layer="24" from-port="0" to-layer="25" to-port="1"/>
		<edge from-layer="25" from-port="2" to-layer="27" to-port="0"/>
		<edge from-layer="26" from-port="0" to-layer="27" to-port="1"/>
		<edge from-layer="27" from-port="2" to-layer="28" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="30" to-port="0"/>
		<edge from-layer="28" from-port="1" to-layer="45" to-port="0"/>
		<edge from-layer="29" from-port="0" to-layer="30" to-port="1"/>
		<edge from-layer="30" from-port="2" to-layer="32" to-port="0"/>
		<edge from-layer="31" from-port="0" to-layer="32" to-port="1"/>
		<edge from-layer="32" from-port="2" to-layer="33" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="35" to-port="0"/>
		<edge from-layer="34" from-port="0" to-layer="35" to-port="1"/>
		<edge from-layer="35" from-port="2" to-layer="37" to-port="0"/>
		<edge from-layer="36" from-port="0" to-layer="37" to-port="1"/>
		<edge from-layer="37" from-port="2" to-layer="38" to-port="0"/>
		<edge from-layer="38" from-port="1" to-layer="40" to-port="0"/>
		<edge from-layer="39" from-port="0" to-layer="40" to-port="1"/>
		<edge from-layer="40" from-port="2" to-layer="42" to-port="0"/>
		<edge from-layer="41" from-port="0" to-layer="42" to-port="1"/>
		<edge from-layer="42" from-port="2" to-layer="43" to-port="0"/>
		<edge from-layer="44" from-port="0" to-layer="45" to-port="1"/>
		<edge from-layer="45" from-port="2" to-layer="47" to-port="0"/>
		<edge from-layer="46" from-port="0" to-layer="47" to-port="1"/>
		<edge from-layer="47" from-port="2" to-layer="48" to-port="0"/>
		<edge from-layer="48" from-port="1" to-layer="50" to-port="0"/>
		<edge from-layer="49" from-port="0" to-layer="50" to-port="1"/>
		<edge from-layer="50" from-port="2" to-layer="52" to-port="0"/>
		<edge from-layer="51" from-port="0" to-layer="52" to-port="1"/>
		<edge from-layer="52" from-port="2" to-layer="53" to-port="0"/>
		<edge from-layer="53" from-port="1" to-layer="55" to-port="0"/>
		<edge from-layer="54" from-port="0" to-layer="55" to-port="1"/>
		<edge from-layer="55" from-port="2" to-layer="57" to-port="0"/>
		<edge from-layer="56" from-port="0" to-layer="57" to-port="1"/>
		<edge from-layer="57" from-port="2" to-layer="58" to-port="0"/>
		<edge from-layer="58" from-port="1" to-layer="59" to-port="0"/>
	</edges>
	<meta_data>
		<MO_version value="2022.1.0-6910-173f328c53d"/>
		<Runtime_version value="2022.1.0-6910-173f328c53d"/>
		<legacy_path value="True"/>
		<cli_parameters>
			<caffe_parser_path value="DIR"/>
			<compress_fp16 value="False"/>
			<data_type value="FP32"/>
			<disable_nhwc_to_nchw value="False"/>
			<disable_omitting_optional value="False"/>
			<disable_resnet_optimization value="False"/>
			<disable_weights_compression value="False"/>
			<enable_concat_optimization value="False"/>
			<enable_flattening_nested_params value="False"/>
			<enable_ssd_gluoncv value="False"/>
			<extensions value="DIR"/>
			<framework value="caffe"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_deprecated_IR_V7 value="False"/>
			<input value="data"/>
			<input_model value="DIR/age_gender_net.caffemodel"/>
			<input_model_is_text value="False"/>
			<input_proto value="DIR/age_gender_net.prototxt"/>
			<input_shape value="[1,3,62,62]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_shape_ops value="True"/>
			<layout value="data(nchw)"/>
			<layout_values value="{'data': {'source_layout': 'nchw', 'target_layout': None, 'is_input': True}}"/>
			<legacy_ir_generation value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'data': {'mean': None, 'scale': array([255.])}}"/>
			<mean_values value="()"/>
			<model_name value="age-gender-recognition-retail-0013"/>
			<output value="['prob', 'age_conv3']"/>
			<output_dir value="DIR"/>
			<packed_user_shapes value="defaultdict(&lt;class 'list'&gt;, {'data': [{'shape': (1, 3, 62, 62), 'port': None, 'added': True}]})"/>
			<placeholder_data_types value="{}"/>
			<placeholder_shapes value="{'data': (1, 3, 62, 62)}"/>
			<progress value="False"/>
			<remove_memory value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="False"/>
			<save_params_from_nd value="False"/>
			<scale_values value="data[255.0]"/>
			<silent value="False"/>
			<source_layout value="()"/>
			<static_shape value="False"/>
			<stream_output value="False"/>
			<target_layout value="()"/>
			<transform value=""/>
			<use_legacy_frontend value="False"/>
			<use_new_frontend value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, input_checkpoint, input_meta_graph, input_symbol, mean_file, mean_file_offsets, move_to_preprocess, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_use_custom_operations_config, transformations_config"/>
		</cli_parameters>
	</meta_data>
</net>
