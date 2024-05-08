import story_processing
import image_generation
import object_localization

import final_image_generation

def main(story_text):
    # 故事处理和提示生成
    prompts = story_processing.generate_prompts(story_text)

    # 对每个场景进行处理
    for scene in prompts:
        # 图像生成
        character_images = image_generation.generate_character_images_with_lora(scene['character_prompts'])

        # 对象定位
        object_bounding_boxes = object_localization.localize_objects_with_grounding_dino(character_images)

        # # 分割掩码获取
        # segmentation_masks = segmentation_mask.get_segmentation_masks_with_sam(object_bounding_boxes)

        # # 边缘检测
        # edges = edge_detection.get_edges_with_pidinet(segmentation_masks)

        # 合成最终图像
        final_image = final_image_generation.generate_final_image_with_dense_conditions(edges, character_images, scene['layout'])

        # 显示或保存最终图像
        # display_or_save_image(final_image)

if __name__ == "__main__":
    story_text = "你的故事文本"
    main(story_text)
