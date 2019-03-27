import torch 
import torch.nn as nn
import torch.nn.functional as F


class GatedTanhLayer(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(GatedTanhLayer, self).__init__()
        self.projrct1 = nn.Linear(in_features=input_dims, out_features=output_dims)
        self.projrct2 = nn.Linear(in_features=input_dims, out_features=output_dims)
    
    def forward(self, input):
        # assert input shape [batch, input_dims]
        y = F.tanh(self.projrct1(input))
        g = F.sigmoid(self.projrct2(input))
        return y * g


class BatchGatedTanhLayer(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(BatchGatedTanhLayer, self).__init__()
        self.projrct1 = nn.Linear(in_features=input_dims, out_features=output_dims)
        self.projrct2 = nn.Linear(in_features=input_dims, out_features=output_dims)
    
    def forward(self, input):
        # assert input shape [batch, input_dims]
        y = F.tanh(self.projrct1(input))
        g = F.sigmoid(self.projrct2(input))
        return y * g


class StepTopDownAttention(nn.Module):
    def __init__(self, d_input, d_attn):
        '''
        d_input: the video feature size 
        d_attn: the attn size used in the attention calculation
        '''
        super(StepTopDownAttention, self).__init__()
        self.w_a = nn.parameter(torch.Tensor(d_attn))
        self.f_a = GatedTanhLayer(d_input, d_attn)
        
    def forward(self, query, key, value, mask=None):
        '''
        step attention function
        query: [batch, d_query]
        key: [batch, n_key, d_feature]
        value: [batch, n_value, d_feature]
        mask: [batch, mask_len]
        '''
        n_key = key.size(1)
        batch = query.size(0)
        # query -> [batch, n_key, d_query]
        query = query.unsqueeze(1).expand(batch, n_key, query.size(2))
        input_tensor = torch.stack([key, query], dim=2)
        attn, context = self.cal_attention(input_tensor, mask)
        return attn, context

    def cal_attention(self, input_tensor, value, mask=None):
        # output: [batch, n_key, d_attn]
        output = self.f_a(input_tensor)
        # W_a -> [batch, 1, d_attn], output -> [batch, d_attn, n_key]
        W_a = self.w_a.expand(output.size(0), 1, output.size(2))
        output = output.permute(0, 2, 1).contiguous()
        score = torch.bmm(W_a, output)
        if mask is not None:
            attention = self.prob_normalize(score, mask.unsqueeze(1))
        else:
            attention = F.softmax(score, dim=-1)
        # attention -> [batch, 1, n_keys], context -> [batch, 1, d_feature]
        context = attention @ value
        return context.squeeze(1), attention.squeeze(1)

    def prob_normalize(self, score, mask):
        """ [(...), T]
        user should handle mask shape"""
        score = score.masked_fill(mask == 0, -1e18)
        norm_score = F.softmax(score, dim=-1)
        return norm_score




