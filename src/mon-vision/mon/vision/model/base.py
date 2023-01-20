class ImageClassificationModel(BaseModel, metaclass=ABCMeta):
    
    def parse_model(
        self,
        d : dict      | None = None,
        ch: list[int] | None = None
    ) -> tuple[Sequential, list[int], list[dict]]:
        """
        Build the model. You have 2 options to build a model: (1) define each
        layer manually, or (2) build model automatically from a config
        dictionary.
        
        We inherit the same idea of model parsing in YOLOv5.
        
        Either way each layer should have the following attributes:
            - i (int): index of the layer.
            - f (int | list[int]): from, i.e., the current layer receive output
              from the f-th layer. For example: -1 means from previous layer;
              -2 means from 2 previous layers; [99, 101] means from the 99th
              and 101st layers. This attribute is used in forward pass.
            - t: type of the layer using this script:
              t = str(m)[8:-2].replace("__main__.", "")
            - np (int): number of parameters using the following script:
              np = sum([x.numel() for x in m.parameters()])
        
        Args:
            d (dict | None): Model definition dictionary. Default to None means
                building the model manually.
            ch (list[int] | None): The first layer's input channels. If given,
                it will be used to further calculate the next layer's input
                channels. Defaults to None means defines each layer in_ and
                out_channels manually.
        
        Returns:
            A Sequential model.
            A list of layer index to save the features during forward pass.
            A list of layer's info (dict) for debugging.
        """
        return parse_model(d=d, ch=ch)
    
    def forward(
        self,
        input  : Tensor,
        augment: bool = False,
        profile: bool = False,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            augment (bool): Perform test-time augmentation. Defaults to False.
            profile (bool): Measure processing time. Defaults to False.
            
        Returns:
            Predictions.
        """
        if augment:
            # For now just forward the input. Later, we will implement the
            # test-time augmentation.
            return self.forward_once(
                input=input, profile=profile, *args, **kwargs
            )
        else:
            return self.forward_once(
                input=input, profile=profile, *args, **kwargs
            )
    
    def show_results(
        self,
        input        : Tensor | None = None,
        target	     : Tensor | None = None,
        pred		 : Tensor | None = None,
        filepath     : foundation.Path | str  | None = None,
        image_quality: int           = 95,
        max_n        : int    | None = 8,
        nrow         : int    | None = 8,
        wait_time    : float         = 0.01,
        save         : bool          = False,
        verbose      : bool          = False,
        *args, **kwargs
    ):
        """
        Show results.

        Args:
            input (Tensor | None): Input.
            target (Tensor | None): Ground-truth.
            pred (Tensor | None): Predictions.
            filepath (foundation.Path | str | None): File path to save the debug result.
            image_quality (int): Image quality to be saved. Defaults to 95.
            max_n (int | None): Show max n images if `image` has a batch size
                of more than `max_n` images. Defaults to None means show all.
            nrow (int | None): The maximum number of items to display in a row.
                The final grid size is (n / nrow, nrow). If None, then the
                number of items in a row will be the same as the number of
                items in the list. Defaults to 8.
            wait_time (float): Wait some time (in seconds) to display the
                figure then reset. Defaults to 0.01.
            save (bool): Save debug image. Defaults to False.
            verbose (bool): If True shows the results on the screen.
                Defaults to False.
        """
        from one.plot import imshow_classification
        
        save_cfg = {
            "filepath"  : filepath or self.debug_image_filepath ,
            "pil_kwargs": dict(quality=image_quality)
        } if save else None
        imshow_classification(
            winname   = self.fullname,  # self.phase.value,
            image     = input,
            pred      = pred,
            target    = target,
            scale     = 2,
            save_cfg  = save_cfg,
            max_n     = max_n,
            nrow      = nrow,
            wait_time = wait_time,
        )


# H2: - Enhancement ------------------------------------------------------------

class ImageEnhancementModel(BaseModel, metaclass=ABCMeta):
    
    def parse_model(
        self,
        d : dict      | None = None,
        ch: list[int] | None = None
    ) -> tuple[Sequential, list[int], list[dict]]:
        """
        Build the model. You have 2 options to build a model: (1) define each
        layer manually, or (2) build model automatically from a config
        dictionary.
        
        We inherit the same idea of model parsing in YOLOv5.
        
        Either way each layer should have the following attributes:
            - i (int): index of the layer.
            - f (int | list[int]): from, i.e., the current layer receive output
              from the f-th layer. For example: -1 means from previous layer;
              -2 means from 2 previous layers; [99, 101] means from the 99th
              and 101st layers. This attribute is used in forward pass.
            - t: type of the layer using this script:
              t = str(m)[8:-2].replace("__main__.", "")
            - np (int): number of parameters using the following script:
              np = sum([x.numel() for x in m.parameters()])
        
        Args:
            d (dict | None): Model definition dictionary. Default to None means
                building the model manually.
            ch (list[int] | None): The first layer's input channels. If given,
                it will be used to further calculate the next layer's input
                channels. Defaults to None means defines each layer in_ and
                out_channels manually.
        
        Returns:
            A Sequential model.
            A list of layer index to save the features during forward pass.
            A list of layer's info (dict) for debugging.
        """
        return parse_model(d=d, ch=ch)
    
    def forward(
        self,
        input  : Tensor,
        augment: bool = False,
        profile: bool = False,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            augment (bool): Perform test-time augmentation. Defaults to False.
            profile (bool): Measure processing time. Defaults to False.
            
        Returns:
            Predictions.
        """
        if augment:
            # For now just forward the input. Later, we will implement the
            # test-time augmentation.
            return self.forward_once(
                input=input, profile=profile, *args, **kwargs
            )
        else:
            return self.forward_once(
                input=input, profile=profile, *args, **kwargs
            )
    
    def show_results(
        self,
        input        : Tensor | None = None,
        target	     : Tensor | None = None,
        pred		 : Tensor | None = None,
        filepath     : foundation.Path | str  | None = None,
        image_quality: int           = 95,
        max_n        : int    | None = 8,
        nrow         : int    | None = 8,
        wait_time    : float         = 0.01,
        save         : bool          = False,
        verbose      : bool          = False,
        *args, **kwargs
    ):
        """
        Show results.

        Args:
            input (Tensor | None): Input.
            target (Tensor | None): Ground-truth.
            pred (Tensor | None): Predictions.
            filepath (foundation.Path | str | None): File path to save the debug result.
            image_quality (int): Image quality to be saved. Defaults to 95.
            max_n (int | None): Show max n images if `image` has a batch size
                of more than `max_n` images. Defaults to None means show all.
            nrow (int | None): The maximum number of items to display in a row.
                The final grid size is (n / nrow, nrow). If None, then the
                number of items in a row will be the same as the number of
                items in the list. Defaults to 8.
            wait_time (float): Wait some time (in seconds) to display the
                figure then reset. Defaults to 0.01.
            save (bool): Save debug image. Defaults to False.
            verbose (bool): If True shows the results on the screen.
                Defaults to False.
        """
        from one.plot import imshow_enhancement

        result = {}
        if input is not None:
            result["input"]  = input
        if target is not None:
            result["target"] = target
        if pred is not None:
            if isinstance(pred, (tuple, list)):
                result["pred"] = pred[-1]
            else:
                result["pred"] = pred
        
        save_cfg = {
            "filepath"  : filepath or self.debug_image_filepath ,
            "pil_kwargs": dict(quality=image_quality)
        } if save else None
        imshow_enhancement(
            winname   = self.fullname,  # self.phase.value,
            image     = result,
            scale     = 2,
            save_cfg  = save_cfg,
            max_n     = max_n,
            nrow      = nrow,
            wait_time = wait_time,
        )
