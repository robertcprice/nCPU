#!/usr/bin/env python3
"""
AUTOREGRESSIVE Division - Predict quotient bits sequentially.

This mimics how CPUs actually do division:
1. Start from MSB of quotient
2. Predict each bit conditioned on previous predictions
3. Use causal masking for the quotient side

This should reach 95%+ accuracy because it matches the computational structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
import sys
import argparse

try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass


class AutoregressiveDivNet(nn.Module):
    """
    Autoregressive division: predict quotient bits one at a time.

    Architecture:
    - Encode dividend and divisor (full visibility)
    - Decode quotient bits autoregressively (causal mask)
    - Cross-attention from quotient to inputs
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_encoder_layers=4, num_decoder_layers=4):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Input embedding for dividend and divisor
        self.input_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Quotient embedding (for autoregressive prediction)
        self.quotient_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Start token embedding
        self.start_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Encoder for dividend/divisor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder for quotient (with causal mask)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

    def generate_causal_mask(self, sz, device):
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, dividend_bits, divisor_bits, quotient_bits=None, teacher_forcing=True):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]
        device = dividend_bits.device

        # Encode inputs
        x = torch.stack([dividend_bits, divisor_bits], dim=-1)
        x = self.input_embed(x)
        x = x + self.pos_embed[:, :bits, :]
        memory = self.encoder(x)

        if teacher_forcing and quotient_bits is not None:
            # Training: use ground truth quotient (shifted right, with start token)
            # Shift quotient right and prepend start token
            q_shifted = quotient_bits[:, :-1].unsqueeze(-1)  # [B, bits-1, 1]
            q_embed = self.quotient_embed(q_shifted)  # [B, bits-1, d_model]

            # Prepend start token
            start = self.start_embed.expand(batch, -1, -1)  # [B, 1, d_model]
            q_embed = torch.cat([start, q_embed], dim=1)  # [B, bits, d_model]
            q_embed = q_embed + self.pos_embed[:, :bits, :]

            # Decode with causal mask
            causal_mask = self.generate_causal_mask(bits, device)
            output = self.decoder(q_embed, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output).squeeze(-1)  # [B, bits]

            return logits
        else:
            # Inference: generate autoregressively
            generated = []
            q_so_far = self.start_embed.expand(batch, -1, -1)  # Start with start token

            for i in range(bits):
                q_with_pos = q_so_far + self.pos_embed[:, :q_so_far.shape[1], :]
                causal_mask = self.generate_causal_mask(q_so_far.shape[1], device)

                output = self.decoder(q_with_pos, memory, tgt_mask=causal_mask)
                logit = self.output_proj(output[:, -1:, :])  # [B, 1, 1]

                pred_bit = (torch.sigmoid(logit) > 0.5).float()
                generated.append(logit.squeeze(-1))  # Store logit for loss

                # Embed prediction and append
                next_embed = self.quotient_embed(pred_bit)  # [B, 1, d_model]
                q_so_far = torch.cat([q_so_far, next_embed], dim=1)

            return torch.cat(generated, dim=1)  # [B, bits]


def generate_div_data(batch_size, bits, device):
    """Generate division data."""
    max_divisor = 2 ** (bits // 2)
    divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)

    max_dividend = 2 ** bits
    dividend = torch.randint(0, max_dividend, (batch_size,), device=device)

    quotient = dividend // divisor

    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_autoregressive(model, device, save_dir, start_bits=8, max_bits=64,
                         batch_size=2048, accum_steps=8):
    """Train autoregressive model."""
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE DIVISION TRAINING")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size} x {accum_steps} = {batch_size * accum_steps}")

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    curriculum = [
        (8, 5000), (12, 8000), (16, 12000), (20, 20000),
        (24, 25000), (28, 30000), (32, 40000),
        (40, 50000), (48, 60000), (56, 70000), (64, 100000)
    ]

    curriculum = [(b, e) for (b, e) in curriculum if b >= start_bits and b <= max_bits]
    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV (AUTOREGRESSIVE)")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1500, T_mult=2, eta_min=5e-6
        )

        best_acc = 0
        consecutive_100 = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            total_loss = 0
            total_correct = 0

            for accum_step in range(accum_steps):
                dividend_bits, divisor_bits, target = generate_div_data(
                    batch_size, bits, device
                )

                with autocast('cuda', dtype=torch.bfloat16):
                    # Teacher forcing during training
                    output = model(dividend_bits, divisor_bits, target, teacher_forcing=True)
                    loss = criterion(output[:, :bits], target[:, :bits])
                    loss = loss / accum_steps

                scaler.scale(loss).backward()
                total_loss += loss.item() * accum_steps

                with torch.no_grad():
                    pred = (torch.sigmoid(output[:, :bits]) > 0.5).float()
                    total_correct += (pred == target[:, :bits]).all(dim=1).float().sum().item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            avg_loss = total_loss / accum_steps
            correct = total_correct / (batch_size * accum_steps) * 100

            if epoch % 100 == 0 or correct > best_acc:
                lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch:5d}: loss={avg_loss:.4f}, acc={correct:.2f}%, lr={lr:.2e}")
                sys.stdout.flush()

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_autoreg_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_autoreg_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--accum-steps', type=int, default=8)
    parser.add_argument('--start-bits', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = AutoregressiveDivNet(
        max_bits=args.bits,
        d_model=384,
        nhead=16,
        num_encoder_layers=4,
        num_decoder_layers=4
    )

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_autoregressive(model, args.device, args.save_dir,
                         start_bits=args.start_bits, max_bits=args.bits,
                         batch_size=args.batch_size, accum_steps=args.accum_steps)


if __name__ == '__main__':
    main()
