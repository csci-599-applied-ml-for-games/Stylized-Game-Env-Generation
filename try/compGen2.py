class CompositeLocalGenerator(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc, prev_output_nc, ngf, n_downsampling, n_blocks_local,
                 use_fg_model=False, no_flow=False,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', scale=1):
        super(CompositeLocalGenerator, self).__init__()
        self.opt = opt
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        self.scale = scale
        activation = nn.ReLU(True)

        if use_fg_model:
            ### individial image generation
            ngf_indv = ngf // 2 if n_downsampling > 2 else ngf
            indv_down = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_indv, kernel_size=7, padding=0),
                         norm_layer(ngf_indv), activation,
                         nn.Conv2d(ngf_indv, ngf_indv * 2, kernel_size=3, stride=2, padding=1),
                         norm_layer(ngf_indv * 2), activation]

            indv_up = []
            for i in range(n_blocks_local):
                indv_up += [
                    ResnetBlock(ngf_indv * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

            indv_up += [
                nn.ConvTranspose2d(ngf_indv * 2, ngf_indv, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_indv), activation]
            indv_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf_indv, output_nc, kernel_size=7, padding=0), nn.Tanh()]

            ### flow and image generation
        ### downsample
        model_down_seg = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                          activation,
                          nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * 2), activation]
        model_down_img = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0),
                          norm_layer(ngf), activation,
                          nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * 2), activation]

        ### resnet blocks
        model_up_img = []
        for i in range(n_blocks_local):
            model_up_img += [
                ResnetBlock(ngf * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

            ### upsample
        up = [nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf),
              activation]
        model_up_img += up
        model_final_img = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        if not no_flow:
            model_up_flow = copy.deepcopy(model_up_img)
            model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]
            model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]

        if use_fg_model:
            self.indv_down = nn.Sequential(*indv_down)
            self.indv_up = nn.Sequential(*indv_up)
            self.indv_final = nn.Sequential(*indv_final)

        self.model_down_seg = nn.Sequential(*model_down_seg)
        self.model_down_img = nn.Sequential(*model_down_img)
        self.model_up_img = nn.Sequential(*model_up_img)
        self.model_final_img = nn.Sequential(*model_final_img)

        if not no_flow:
            self.model_up_flow = nn.Sequential(*model_up_flow)
            self.model_final_flow = nn.Sequential(*model_final_flow)
            self.model_final_w = nn.Sequential(*model_final_w)

    def forward(self, input, img_prev, mask, img_feat_coarse, flow_feat_coarse, img_fg_feat_coarse, use_raw_only):
        flow_multiplier = 20 * (2 ** self.scale)
        down_img = self.model_down_seg(input) + self.model_down_img(img_prev)
        img_feat = self.model_up_img(down_img + img_feat_coarse)
        img_raw = self.model_final_img(img_feat)

        flow = weight = flow_feat = None
        if not self.no_flow:
            down_flow = down_img
            flow_feat = self.model_up_flow(down_flow + flow_feat_coarse)
            flow = self.model_final_flow(flow_feat) * flow_multiplier
            weight = self.model_final_w(flow_feat)

        gpu_id = img_feat.get_device()
        if use_raw_only or self.no_flow:
            img_final = img_raw
        else:
            img_warp = self.resample(img_prev[:, -3:, ...].cuda(gpu_id), flow).cuda(gpu_id)
            weight_ = weight.expand_as(img_raw)
            img_final = img_raw * weight_ + img_warp * (1 - weight_)

        img_fg_feat = None
        if self.use_fg_model:
            img_fg_feat = self.indv_up(self.indv_down(input) + img_fg_feat_coarse)
            img_fg = self.indv_final(img_fg_feat)
            mask = mask.cuda(gpu_id).expand_as(img_raw)
            img_final = img_fg * mask + img_final * (1 - mask)
            img_raw = img_fg * mask + img_raw * (1 - mask)

        return img_final, flow, weight, img_raw, img_feat, flow_feat, img_fg_feat
