solve0 ：无cache，根据一阶梯度进行变量选取，即选maxmin
solve：cache+一阶梯度进行变量选取
solve2：cache+根据二阶梯度选取，libsvm中方法	+shrinking（只去除α=0的样本，不去α=c，无梯度重建）
	shrinking后cache可存列数会增多，未实现利用，
	负载平衡（-m -n值会影响通信开销？未解决）