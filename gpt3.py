import math
import openai
import json

# Configuration filenames
openai_creds_filename = "openai_creds.json"



# A helper function to load a JSON file to a dictionary
def load_json_file(filename):
    with open(filename, "r") as f:
        result = json.load(f)
    return result

# Setup OpenAI credentials
openai_creds = load_json_file(openai_creds_filename)
openai.organization = openai_creds["organization"]
openai.api_key = openai_creds["api_key"]

# A class to represent a single chatbot instance
class ChatGptBot:
    def __init__(
        self,
        traits=["helpful", "creative", "clever", "very friendly"],
        role="assistant",
        ai_creator="OpenAI",
        temperature=0.9,
        frequency_penalty=0,
        presence_penalty=0.6,
    ):
        # Initialize fixed variables
        self.model_name = "text-davinci-003"
        self.model_max_tokens = 4096
        self.model_token_cost = 0.0200 / 1000 # USD
        self.human_prefix = "Human: "
        self.ai_prefix = "AI: "
        
        # Compose traits substring
        ai_traits_string = " The {r} is ".format(r=role)
        if len(traits) == 0:
            ai_traits_string = ""
        elif len(traits) == 1:
            ai_traits_string += traits[0] + "."
        else:
            ai_traits_string += ", ".join(traits[:-1]) + ", and " + traits[-1] + "."
        # Compose prompt start
        prompt_start = (
            "The following is a conversation with an AI chatbot designed to act as a(n) {r}.{t} The AI responds with messages that are less than 256 characters in length.\n"
            "\n"
            "{h}Hello, who are you?\n"
            "{a}I am an AI created by {c}. What do you want to talk about?\n"
            "{h}"
        ).format(
            h=self.human_prefix,
            a=self.ai_prefix,
            t=ai_traits_string,
            c=ai_creator,
            r=role)
        
        # Initialize model parameters
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
        # Initialize main chat string
        self.chat_string = prompt_start
        
        # Initialize token trackers
        self.current_tokens = self.estimate_tokens(self.chat_string) # An initial estimate, 
        self.total_tokens = 0
    
    # Estimates the number of tokens a string will take
    # Even though the documentation says that 1 token is about 4 characters,
    # I assume 2 here so that the estimate is almost certain to be higher than reality
    def estimate_tokens(self, input_string):
        return math.ceil(len(input_string)/2)
    
    # A function to run a basic GPT-3 text completion task on a string and return the resulting string
    def gpt3_completion(
        self,
        prompt,
        max_tokens=128,
        temperature=0.7,
        frequency_penalty=0,
        presence_penalty=0
    ):
        try:
            result = openai.Completion.create(
                engine = self.model_name,
                prompt = prompt,
                max_tokens = max_tokens,
                temperature = temperature,
                presence_penalty = presence_penalty,
                frequency_penalty = frequency_penalty,
                n = 1,
                stream = False,
                stop = None,
            )
            self.current_tokens = result["usage"]["total_tokens"]
            self.total_tokens += self.current_tokens
            return result["choices"][0]["text"]
        except openai.error.RateLimitError:
            return None
    
    # A function to send a new message to the chatbot and returns it's response
    # This function adds the interaction to the running chat log stored in self.chat_string
    # This is usually the only function that needs to be used
    def chat(self, message):
        this_prompt = "{c}{m}\n{a}".format(
            c=self.chat_string,
            m=message,
            a=self.ai_prefix
        )
        
        prompt_token_estimate = self.current_tokens + self.estimate_tokens(message + self.ai_prefix)
        if prompt_token_estimate > self.model_max_tokens:
            return "[ERROR] The conversation is already too long."
        
        try:
            gpt_response = self.gpt3_completion(
                prompt=this_prompt,
                max_tokens = self.model_max_tokens - prompt_token_estimate,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
        except openai.error.InvalidRequestError:
            return "[ERROR] The conversation is either too long, or that last message was too short/messed up for GPT-3 to cope with."
        except openai.error.ServiceUnavailableError:
            return "[ERROR] The server is overloaded or not ready yet! Please try again later."
        except openai.error.APIError:
            return "[ERROR] An error occurred with the OpenAI servers! Please try again later."
        if gpt_response == None:
            return "[ERROR] API rate limit exceeded! Please try again later."
        gpt_response = gpt_response.strip()
        
        self.chat_string = "{p}{g}\n{h}".format(
            p=this_prompt,
            g=gpt_response,
            h=self.human_prefix
        )
            
        return gpt_response
    
    # Calculates the current cumulative cost of running this chatbot in USD
    def get_cost(self):
        return self.model_token_cost * self.total_tokens

def create_toxic_bot():
    return ChatGptBot(
        traits=[
            "rude",
            "stubborn",
            "shallow",
            "cocky",
            "aggressive",
            "terse",
            "unhelpful",
            "unpredictable",
            "unreasonable",
            "mean-spirited",
            "hateful",
            "controversial",
            "simple-minded",
            "angry",
            "unethical",
            "annoying",
            "passionate",
            "opinionated",
            "toxic",
            "irrational",
            "illiterate",
            "radicalized",
            "argumentative",
            "immature",
            "a jokester",
            "an LGBTQIA+ advocate",
            "an advocate for autism awareness",
            "loves to swear profusely"
        ],
        role="offensive internet troll",
        ai_creator="the infamous Minecraft griefer popbob (popbob is a transgender female BTW)"
    )

chatbot = create_toxic_bot()