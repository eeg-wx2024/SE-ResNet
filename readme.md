09/20/1637
    train_N = 1
    val_N = 1


09/20/1640
    train_N = 2
    val_N = 1


# EEGChannelNet
0.1   12/02/2147          8.05   8.30   9.54  9.88
0.2   12/03/1525
0.3   12/03/1527
0.4   12/03/1528/18
0.5   12/03/1528/41
0.6   12/03/1529/09
0.7   12/03/1529/40
0.8   12/03/1531
0.9   12/03/1532
1.0   12/03/1533

# 修改点
# 1、MT（Merget training）
  train_N = 2   # 训练时两个合并为一个进行训练
  val_N = 1

# 2、ADCT   main.py
main.py    x = x_group(x) # ctrl+单击"x_group"
data_utils.py   x = group_replace_patch(x, ratio=0.25)
  
# 3、ELU
class GroupResNet_noscore(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_model = resnet(attn_name="se", act_name="elu", blocks=[2, 2, 2, 2])
        # elu,relu,silu
        # "se" "cbam""ca"

# 4、attention
class GroupResNet_noscore(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_model = resnet(attn_name="se", act_name="elu", blocks=[2, 2, 2, 2])
        # elu,relu,silu
        # "se" "cbam""ca"
# 5、vote  main.py
  x = to_patch(x, replace=True)  #拼图
  x = x_group(x)                 #分组拼图投票