from tensorflow.keras import losses
import tensorflow as tf
import itertools
from typing import Any, Optional

_EPSILON = tf.keras.backend.epsilon()

@tf.keras.utils.register_keras_serializable()
class BinaryBoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 1,
                 boundary_alpha: float = 2., **kwargs):
        """
        Args:
            BoundaryLoss is the sum of semantic segmentation loss.
            The BoundaryLoss loss is a binary cross entropy loss.
            
            from_logits       (bool)  : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool)  : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)   : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)   : Number of classes to classify (must be equal to number of last filters in the model)
            boundary_alpha    (float) : Boundary loss alpha
        """
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
        self.boundary_alpha = boundary_alpha
    
        if self.use_multi_gpu:
            self.loss_reduction = losses.Reduction.NONE
        else:
            # self.loss_reduction = losses.Reduction.AUTO
            self.loss_reduction = losses.Reduction.NONE

    def get_config(self):
        config = super().get_config()
        config.update(from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # Calc bce loss                
        edge_map = tf.cast(y_true, dtype=tf.float32)
        grad_components = tf.image.sobel_edges(edge_map)
        grad_mag_components = grad_components ** 2

        grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)

        edge = tf.sqrt(grad_mag_square)
        edge = tf.cast(tf.where(edge>=0.1, 1., 0.), dtype=tf.float32)

        loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=self.from_logits, reduction=self.loss_reduction)(y_true=edge, y_pred=y_pred)

        # Reduce loss to scalar
        # if self.use_multi_gpu:
        loss = tf.reduce_mean(loss)
        
        loss *= self.boundary_alpha
        return loss


@tf.keras.utils.register_keras_serializable()
class BinaryAuxiliaryLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 1,
                 aux_alpha: float = 0.4,
                  **kwargs):
        """
        Args:
            AuxiliaryLoss is the sum of semantic segmentation loss.
            The AuxiliaryLoss loss is a cross entropy loss.
              
            from_logits       (bool)  : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool)  : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)   : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)   : Number of classes to classify (must be equal to number of last filters in the model)
            aux_alpha         (float) : Aux loss alpha.
        """
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
        self.aux_alpha = aux_alpha
        
        if self.use_multi_gpu:
            self.loss_reduction = losses.Reduction.NONE
        else:
            # self.loss_reduction = losses.Reduction.AUTO
            self.loss_reduction = losses.Reduction.NONE

    def get_config(self):
        config = super().get_config()
        config.update(from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config


    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits, reduction=self.loss_reduction)(y_true=y_true, y_pred=y_pred)
        
        # if self.use_multi_gpu:
        loss = tf.reduce_mean(loss)

        loss *= self.aux_alpha
        return loss


@tf.keras.utils.register_keras_serializable()
class HumanSegLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, class_weight: Optional[Any] = None,
                 from_logits: bool = True, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 3,
                 dataset_name: str = 'cityscapes',
                 loss_type: str = 'focal',
                  **kwargs):
        """
        Args:
            SemanticLoss is the sum of semantic segmentation loss and confidence loss.
            The semantic loss is a sparse categorical loss,
            and the confidence loss is calculated as binary cross entropy.
              
            gamma             (float): Focal loss's gamma.
            class_weight      (Array): Cross-entropy loss's class weight (logit * class_weight)
            from_logits       (bool) : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool) : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)  : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)  : Number of classes to classify (must be equal to number of last filters in the model)
            dataset_type      (str)  : Train dataset type. For Cityscapes, the process of excluding ignore labels is included.
            loss_type         (str)  : Train loss function type.
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.loss_type = loss_type
        self.smooth = 0.00001


        if self.use_multi_gpu:
            self.loss_reduction = losses.Reduction.NONE
        else:
            # self.loss_reduction = losses.Reduction.AUTO
            self.loss_reduction = losses.Reduction.NONE


    def get_config(self):
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.class_weight,
                      from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config


    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # # BCE loss
        # bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=self.from_logits, reduction=self.loss_reduction)(y_true=y_true, y_pred=y_pred)

        # if self.use_multi_gpu:
        #     bce_loss = tf.reduce_mean(bce_loss)

        # # Dice loss
        # y_true_f = tf.keras.backend.flatten(y_true)
        # y_pred_f = tf.keras.backend.flatten(y_pred)
        # intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        # dice = (2. * intersection + 100) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 100)
        # dice_loss = 1 - dice

        # if self.use_multi_gpu:
        #     dice_loss = dice_loss * (1. / self.global_batch_size)

        # loss = (dice_loss * 0.5) + (bce_loss * 0.5)

        # Semantic loss
        # loss = self.sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred, gamma=self.gamma, from_logits=self.from_logits)
        loss = self.sparse_categorical_cross_entropy(y_true=y_true, y_pred=y_pred)

        return loss


    def sparse_categorical_cross_entropy(self, y_true, y_pred):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=self.loss_reduction)(y_true=y_true, y_pred=y_pred)
             
        # if self.use_multi_gpu:
        loss = tf.reduce_mean(loss)
        
        return loss
        
    
    def sparse_categorical_focal_loss(self, y_true, y_pred, gamma, *,
                                  class_weight: Optional[Any] = None,
                                  from_logits: bool = False, axis: int = -1,
                                  ) -> tf.Tensor:
        # Process focusing parameter
        gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
        gamma_rank = gamma.shape.rank
        scalar_gamma = gamma_rank == 0

        # Process class weight
        if class_weight is not None:
            class_weight = tf.convert_to_tensor(class_weight,
                                                dtype=tf.dtypes.float32)

        # Process prediction tensor
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred_rank = y_pred.shape.rank
        if y_pred_rank is not None:
            axis %= y_pred_rank
            if axis != y_pred_rank - 1:
                # Put channel axis last for sparse_softmax_cross_entropy_with_logits
                perm = list(itertools.chain(range(axis),
                                            range(axis + 1, y_pred_rank), [axis]))
                y_pred = tf.transpose(y_pred, perm=perm)
        elif axis != -1:
            raise ValueError(
                f'Cannot compute sparse categorical focal loss with axis={axis} on '
                'a prediction tensor with statically unknown rank.')
        y_pred_shape = tf.shape(y_pred)

        # Process ground truth tensor
        y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
        y_true_rank = y_true.shape.rank

        if y_true_rank is None:
            raise NotImplementedError('Sparse categorical focal loss not supported '
                                    'for target/label tensors of unknown rank')

        reshape_needed = (y_true_rank is not None and y_pred_rank is not None and
                        y_pred_rank != y_true_rank + 1)
        if reshape_needed:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

        if from_logits:
            logits = y_pred
            probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            probs = y_pred
            logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))
        
        xent_loss = losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            reduction=self.loss_reduction)(y_true=y_true, y_pred=logits)

        y_true_rank = y_true.shape.rank
        probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)

        if not scalar_gamma:
            gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)
        focal_modulation = (1 - probs) ** gamma

        loss = focal_modulation * xent_loss

        if self.use_multi_gpu:
            loss = tf.reduce_mean(loss)

        if class_weight is not None:
            class_weight = tf.gather(class_weight, y_true, axis=0,
                                    batch_dims=y_true_rank)
            loss *= class_weight

        # if reshape_needed:
        #     print('y_pred_shape', y_pred_shape)
        #     print('loss_shape', loss)
        #     loss = tf.reshape(loss, y_pred_shape[:-1])

        return loss