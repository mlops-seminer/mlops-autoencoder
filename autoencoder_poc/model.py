import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_channels, input_size, enc_channels, latent_dim):
        super(Autoencoder, self).__init__()

        if int(input_channels) <= 0:
            raise ValueError("input_channels は1以上で指定してください。")
        if len(input_size) != 2 or any(int(size) <= 0 for size in input_size):
            raise ValueError("input_size は正の整数2要素で指定してください。")
        if not enc_channels or any(int(ch) <= 0 for ch in enc_channels):
            raise ValueError("enc_channels は正の整数を1個以上指定してください。")
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim は1以上で指定してください。")

        downsample_factor = 2 ** len(enc_channels)
        if any(int(size) % downsample_factor != 0 for size in input_size):
            raise ValueError(
                "input_size の縦横は、2 ** len(enc_channels) "
                f"(={downsample_factor}) で割り切れる必要があります: {list(input_size)}"
            )
        
        # --- Encoder ---
        self.encoder_layers = nn.ModuleList()
        in_ch = input_channels
        
        # 画像サイズの変化を追跡
        current_h, current_w = input_size
        
        for ch in enc_channels:
            # Conv(3x3) -> ReLU -> MaxPool(2x2)
            # これにより空間サイズは半分になる
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_ch = ch
            current_h //= 2
            current_w //= 2
            
        self.flatten_dim = in_ch * current_h * current_w
        self.latent_h = current_h
        self.latent_w = current_w
        self.last_conv_ch = in_ch
        
        # ボトルネック
        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)
        
        # --- Decoder ---
        # ボトルネックからの復元
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder_layers = nn.ModuleList()
        
        # Encoderの逆順でチャネルを設定: 例 [128, 64, 32]
        reversed_channels = enc_channels[::-1] 
        
        # Decoderの各層の入力チャネル初期値
        dec_in_ch = reversed_channels[0]
        
        for i in range(len(reversed_channels)):
            # 出力チャネルを決める
            if i < len(reversed_channels) - 1:
                # 次の層（より浅い層）のチャネル数へ
                dec_out_ch = reversed_channels[i+1]
                
                # ConvTranspose2d(k=2, s=2) でサイズを2倍にする
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(dec_in_ch, dec_out_ch, kernel_size=2, stride=2),
                        nn.BatchNorm2d(dec_out_ch),
                        nn.ReLU(),
                        # 精細化のためのConv層を追加しても良いが、シンプルにするため省略
                        # nn.Conv2d(dec_out_ch, dec_out_ch, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(dec_out_ch),
                        # nn.ReLU()
                    )
                )
            else:
                # 最終層：元の入力チャネル数に戻す
                dec_out_ch = input_channels
                
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(dec_in_ch, dec_out_ch, kernel_size=2, stride=2),
                        nn.Sigmoid() # 入力画像は0-1に正規化されている前提
                    )
                )
            
            dec_in_ch = dec_out_ch

    def forward(self, x):
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Flatten
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        
        # Decoder
        x = self.fc_dec(z)
        # Unflatten
        x = x.view(x.size(0), self.last_conv_ch, self.latent_h, self.latent_w)
        
        for layer in self.decoder_layers:
            x = layer(x)
            
        return x
