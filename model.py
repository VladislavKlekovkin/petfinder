import torch
import timm


sigmoid_torch = torch.nn.Sigmoid()


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid_torch(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = sigmoid_torch(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class SwishModule(torch.nn.Module):

    def forward(self, x):
        return Swish.apply(x)


class Model(torch.nn.Module):

    def __init__(self, kernel_type, n_meta_features=0, n_meta_dim=[512, 128], **kwargs):

        super().__init__()
        self.n_meta_features = n_meta_features
        if 'swin_' in kernel_type:
            self.model = timm.create_model(kernel_type, pretrained=True)
        elif 'efficientnet' in kernel_type:
            self.model = timm.create_model(kernel_type, pretrained=True)#,drop_rate=kwargs['drop_rate'], drop_path_rate=kwargs['drop_path_rate'])

        for param in self.model.parameters():
            param.requires_grad = False

        self.dropouts = torch.nn.ModuleList([
            torch.nn.Dropout(0.5) for _ in range(5)
        ])

        if 'swin_' in kernel_type:
            in_features = self.model.head.in_features
            self.model.head = torch.nn.Identity()
        elif 'efficientnet' in kernel_type:
            in_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Identity()

        if n_meta_features > 0:
            self.meta = torch.nn.Sequential(
                torch.nn.Linear(n_meta_features, n_meta_dim[0]),
                torch.nn.BatchNorm1d(n_meta_dim[0]),
                SwishModule(),
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                torch.nn.BatchNorm1d(n_meta_dim[1]),
                SwishModule(),
            )
            in_features += n_meta_dim[1]

        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x, x_meta=None):

        x = self.model(x)

        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.linear(dropout(x))
            else:
                out += self.linear(dropout(x))

        out /= len(self.dropouts)

        return out

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
