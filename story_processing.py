from prompt import *
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


# 调用openai的api，生成故事
# 根据故事生成的流程 1. Story Generation 2. Panel Split 3. Global Prompt Generation 4. Layout Generation。 进行生成，并且保存到文件中

class StoryToImagePipeline:
    def __init__(self, openai_api_key):
        self.token_id = 'token_id'
        self.llm = OpenAI(openai_api_key=openai_api_key)

    # def complete_prompt(self, prompt):
    #     response = self.client.create(
    #         model="gpt-3.5-turbo",
    #         prompt=prompt,
    #         max_tokens=1024  # Adjust as necessary
    #     )
    #     return response.choices[0].text

    # 1. 故事生成
    def story_generation(self, initial_prompt):
        gen_prompt = PromptTemplate.from_template(prompt_Story_generation).format(story_request=initial_prompt)
        return self.llm.invoke(gen_prompt)

    # 2. 面板切分
    def panel_gen(self, story_text):
        # 定义分割故事的逻辑
        split_prompt = PromptTemplate.from_template(prompt_Panel_Split).format(story=story_text)
        return self.llm.invoke(split_prompt)

    def panel_split(story_text):
    # 使用正则表达式来匹配面板的开始
        panel_pattern = r"Panel \d+:"
        panels = re.split(panel_pattern, story_text)

        # 删除空字符串并去除前后空格
        panels = [panel.strip() for panel in panels if panel.strip()]

        return panels
    # 3. 全局提示生成
    def global_prompt_generation(self, panel_text):
        # 定义生成全局提示的逻辑
        global_prompt = PromptTemplate.from_template(prompt_Global_Prompt_Generation).format(panel=panel_text)
        return self.llm.invoke(global_prompt)
    

    # 4. 布局生成
    def layout_generation(self, global_prompt):
        # 定义生成布局的逻辑
        # 生成的内容
        '''Caption: Illustrate a serene scene with Chisato andFujiwara resting together, enjoying the tranquility oftheir surroundings in the forest.Objects: [('Chisato, admiring the beautiful scenery',[164, 61, 261, 448]), ('Fujiwara, admiring thebeautiful scenery', [431, 47, 331, 460])]Background prompt: In the forest'''
        layout_prompt = PromptTemplate.from_template(prompt_Layou_Generation).format(global_prompt=global_prompt)
        return self.llm.invoke(layout_prompt)
    
    def pipeline(self, initial_prompt):
        story_text = self.story_generation(initial_prompt)
        panels = self.panel_gen(story_text)
        panels = self.panel_split(panels)
        layouts = []
        global_prompts = []
        for panel in panels:
            global_prompt = self.global_prompt_generation(panel)
            global_prompts.append(global_prompt)
            layout = self.layout_generation(global_prompt)
            layouts.append(layout)
        return global_prompts,layouts
    
# 使用示例
if __name__ == '__main__':
    api_key = "你的OpenAI API密钥"
    pipeline = StoryToImagePipeline(api_key)
    story_text = pipeline.story_generation("初始故事提示")
    panels = pipeline.panel_gen(story_text)
    panels = pipeline.panel_split(panels)
    for panel in panels:
        global_prompt = pipeline.global_prompt_generation(panel)
        layout = pipeline.layout_generation(global_prompt)