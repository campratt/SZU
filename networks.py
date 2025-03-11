import torch
from torch import nn
import torch.nn.functional as F


    
class DoubleConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock3D, self).__init__()
        self.conv_res = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), padding=(0,0,0))
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        

    def forward(self, x):
        x_res = self.conv_res(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        out = x + x_res
        
        return out



class AttentionBlock3D(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_l, F_g, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock3D, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv3d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip_connection, gate, return_attn=False):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        if return_attn:
            return out, psi
        else:
            return out
        


class NestedUNet3D(nn.Module):
    def __init__(self, input_channels=9, model_size='large'):
        super().__init__()

        if model_size == 'small':
            nb_filter = [8, 16, 32, 64, 128]
        elif model_size == 'large':
            nb_filter = [16, 32, 64, 128, 256]


        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)

        self.conv0_0 = DoubleConvBlock3D(1, nb_filter[0])
        self.conv1_0 = DoubleConvBlock3D(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConvBlock3D(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConvBlock3D(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConvBlock3D(nb_filter[3], nb_filter[4])

        self.attn0_1 = AttentionBlock3D(F_l=nb_filter[0], F_g=nb_filter[1], n_coefficients=nb_filter[0])
        self.attn1_1 = AttentionBlock3D(F_l=nb_filter[1], F_g=nb_filter[2], n_coefficients=nb_filter[1])
        self.attn2_1 = AttentionBlock3D(F_l=nb_filter[2], F_g=nb_filter[3], n_coefficients=nb_filter[2])
        self.attn3_1 = AttentionBlock3D(F_l=nb_filter[3], F_g=nb_filter[4], n_coefficients=nb_filter[3])

        self.attn0_2 = AttentionBlock3D(F_l=nb_filter[0], F_g=nb_filter[1], n_coefficients=nb_filter[0])
        self.attn1_2 = AttentionBlock3D(F_l=nb_filter[1], F_g=nb_filter[2], n_coefficients=nb_filter[1])
        self.attn2_2 = AttentionBlock3D(F_l=nb_filter[2], F_g=nb_filter[3], n_coefficients=nb_filter[2])

        self.attn0_3 = AttentionBlock3D(F_l=nb_filter[0], F_g=nb_filter[1], n_coefficients=nb_filter[0])
        self.attn1_3 = AttentionBlock3D(F_l=nb_filter[1], F_g=nb_filter[2], n_coefficients=nb_filter[1])

        self.attn0_4 = AttentionBlock3D(F_l=nb_filter[0], F_g=nb_filter[1], n_coefficients=nb_filter[0])

        self.conv0_1 = DoubleConvBlock3D(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConvBlock3D(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConvBlock3D(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConvBlock3D(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConvBlock3D(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConvBlock3D(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConvBlock3D(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConvBlock3D(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConvBlock3D(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConvBlock3D(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])


        
        self.final_map1 = nn.Conv3d(4 * nb_filter[0], 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn_final_map1 = nn.BatchNorm3d(1)
        self.leaky_relu = nn.LeakyReLU()
        self.final_map2 = nn.Conv2d(input_channels, 1, kernel_size=(3,3), padding=(1,1))



    def forward(self, input, return_attn=False):
        input = input.unsqueeze(1)

        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0_up = self.up(x1_0)
        attn0_1, psi0_1 = self.attn0_1(x0_0, x1_0_up, return_attn=True)
        x0_1 = self.conv0_1(torch.cat([x1_0_up, attn0_1], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0_up = self.up(x2_0)
        attn1_1, psi1_1 = self.attn1_1(x1_0, x2_0_up, return_attn=True)
        x1_1 = self.conv1_1(torch.cat([x2_0_up, attn1_1], 1))

        x1_1_up = self.up(x1_1)
        attn0_2, psi0_2 = self.attn0_2(x0_1, x1_1_up, return_attn=True)
        x0_2 = self.conv0_2(torch.cat([x1_1_up, attn0_1, attn0_2], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0_up = self.up(x3_0)
        attn2_1, psi2_1 = self.attn2_1(x2_0, x3_0_up, return_attn=True)
        x2_1 = self.conv2_1(torch.cat([x3_0_up, attn2_1], 1))

        x2_1_up = self.up(x2_1)
        attn1_2, psi1_2 = self.attn1_2(x1_1, x2_1_up, return_attn=True)
        x1_2 = self.conv1_2(torch.cat([x2_1_up, attn1_1, attn1_2], 1))

        x1_2_up = self.up(x1_2)
        attn0_3, psi0_3 = self.attn0_3(x0_2, x1_2_up, return_attn=True)
        x0_3 = self.conv0_3(torch.cat([x1_2_up, attn0_1, attn0_2, attn0_3], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_up = self.up(x4_0)
        attn3_1, psi3_1 = self.attn3_1(x3_0, x4_0_up, return_attn=True)
        x3_1 = self.conv3_1(torch.cat([x4_0_up, attn3_1], 1))

        x3_1_up = self.up(x3_1)
        attn2_2, psi2_2 = self.attn2_2(x2_1, x3_1_up, return_attn=True)
        x2_2 = self.conv2_2(torch.cat([x3_1_up, attn2_1, attn2_2], 1))

        x2_2_up = self.up(x2_2)
        attn1_3, psi1_3 = self.attn1_3(x1_2, x2_2_up, return_attn=True)
        x1_3 = self.conv1_3(torch.cat([x2_2_up, attn1_1, attn1_2, attn1_3], 1))

        x1_3_up = self.up(x1_3)
        attn0_4, psi0_4 = self.attn0_4(x0_3, x1_3_up, return_attn=True)
        x0_4 = self.conv0_4(torch.cat([x1_3_up, attn0_1, attn0_2, attn0_3, attn0_4], 1))


        
        
        x = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        x = self.final_map1(x)
        x = x[:, 0]
        out_map = self.final_map2(x)[:, 0]
        

        if return_attn:
            return out_map, [psi0_1, psi1_1, psi2_1, psi3_1, psi0_2, psi1_2, psi2_2, psi0_3, psi1_3, psi0_4]
        else:
            return out_map
        


