import math
import openai
import json
from enum import Enum
import tenacity as tc
import javascript as js
mineflayer = js.require('mineflayer')

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
        max_response_length=None, # May not be respected in all cases
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
        ai_traits_string = " The AI is "
        if len(traits) == 0:
            ai_traits_string = ""
        elif len(traits) == 1:
            ai_traits_string += traits[0] + "."
        else:
            ai_traits_string += ", ".join(traits[:-1]) + ", and " + traits[-1] + "."
        # Compose response length substring
        length_substring = " The AI always responds with messages that are strictly less than {l} characters long."
        if max_response_length == None:
            length_substring = ""
        else:
            if max_response_length <= 0:
                length_substring = ""
            else:
                length_substring = length_substring.format(l=max_response_length)
        # Compose prompt start
        prompt_start = (
            "The following is a conversation with an AI chatbot who is a(n) {r}.{t}{l}\n"
            "\n"
            "{h}Hello, who are you?\n"
            "{a}I am an AI created by {c}. What do you want to talk about?\n"
            "{h}"
        ).format(
            h=self.human_prefix,
            a=self.ai_prefix,
            t=ai_traits_string,
            c=ai_creator,
            r=role,
            l=length_substring)
        
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
    
    class ErrorCode(Enum):
        ERR_NONE = 0
        ERR_CONVO_TOO_LONG = 1
        ERR_INVALID_REQUEST = 2
        ERR_SERVICE_UNAVAILABLE = 3
        ERR_API = 4
        ERR_RATE_LIMIT = 5
        ERR_RETRY_FAIL = 6
    
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
    
    # A function to send a new message to the chatbot and returns it's response plus an error code
    # This function adds the interaction to the running chat log stored in self.chat_string
    # This is prone to failures that can be fixed with retries
    def chat(self, message):
        this_prompt = "{c}{m}\n{a}".format(
            c=self.chat_string,
            m=message,
            a=self.ai_prefix
        )
        
        prompt_token_estimate = self.current_tokens + self.estimate_tokens(message + self.ai_prefix)
        if prompt_token_estimate > self.model_max_tokens:
            return ("[ERROR] The conversation is already too long.", self.ErrorCode.ERR_CONVO_TOO_LONG)
        
        try:
            gpt_response = self.gpt3_completion(
                prompt=this_prompt,
                max_tokens = self.model_max_tokens - prompt_token_estimate,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
        except openai.error.InvalidRequestError:
            return ("[ERROR] The conversation is either too long, or that last message was too messed up for GPT-3 to cope with.", self.ErrorCode.ERR_INVALID_REQUEST)
        except openai.error.ServiceUnavailableError:
            return ("[ERROR] The server is overloaded or not ready yet! Please try again later.", self.ErrorCode.ERR_SERVICE_UNAVAILABLE)
        except openai.error.APIError:
            return ("[ERROR] An error occurred with the OpenAI servers! Please try again later.", self.ErrorCode.ERR_API)
        if gpt_response == None:
            return ("[ERROR] API rate limit exceeded! Please try again later.", self.ErrorCode.ERR_RATE_LIMIT)
        gpt_response = gpt_response.strip()
        
        self.chat_string = "{p}{g}\n{h}".format(
            p=this_prompt,
            g=gpt_response,
            h=self.human_prefix
        )
            
        return (gpt_response, self.ErrorCode.ERR_NONE)
    
    # A function to use the chat() function in a more reliable manor, with fewer failures
    # This function uses random exponential backoff to retry the chat() command for errors that can be fixed with retries
    # It also attempts to use some workarounds to tweak the input message to be acceptable for the OpenAI API
    # It has a limit on how many times to retry the function
    def chat_retry(self, message, max_tries=6):
        @tc.retry(wait=tc.wait_random_exponential(min=1, max=60), stop=tc.stop_after_attempt(max_tries))
        def chat_with_backoff(in_message):
            response, error = self.chat(in_message)
            
            # If the message may have been too short, try lengthening it with spaces
            if error == self.ErrorCode.ERR_INVALID_REQUEST:
                in_message += " "*16
                # Manually retry it now
                response, error = self.chat(in_message)
                # If it still has the same error, assume it's hopeless and return
                if error == self.ErrorCode.ERR_INVALID_REQUEST:
                    return (response, error)
            
            # If it was a normal success, return
            if error == self.ErrorCode.ERR_NONE:
                return (response, error)
            
            # If there is no chance of a retry or modification helping, return
            if error in [
                self.ErrorCode.ERR_CONVO_TOO_LONG
            ]:
                return (response, error)
            
            # If the issue is definitely temporary, raise an exception so that it can be retried
            if error in [
                self.ErrorCode.ERR_SERVICE_UNAVAILABLE,
                self.ErrorCode.ERR_API,
                self.ErrorCode.ERR_RATE_LIMIT
            ]:
                raise SystemError("Failed to get a response from GPT-3. Please try again later.")
            
        # Run it
        try:
            response, error = chat_with_backoff(message)
        except SystemError:
            response = "[ERROR] Failed to get a response from GPT-3 after {} attempts. Please try again later.".format(max_tries)
            error = self.ErrorCode.ERR_RETRY_FAIL
        return (response, error)
    
    # A dummy function that basically repeats the input message
    # Useful for testing the chatbot without actually doing text completion
    def chat_dummy(self, message):
        return "Only dumb people say \"{}\", lol.".format(message)
    
    # Calculates the current cumulative cost of running this chatbot in USD
    def get_cost(self):
        return self.model_token_cost * self.total_tokens
    
    # Get variable functions
    def get_chat_log(self):
        return self.chat_string.rstrip(self.human_prefix)





def create_toxic_bot():
    return ChatGptBot(
        traits=[
            "rude",
            "stubborn",
            "shallow",
            "arrogant",
            "cocky",
            "aggressive",
            "adversarial",
            "terse",
            "unhelpful",
            "unlikable",
            "disagreeable",
            "unpredictable",
            "unreasonable",
            "mean-spirited",
            "hateful",
            "inhumane",
            "conceited",
            "abrasive",
            "controversial",
            "combative",
            "simple-minded",
            "angry",
            "unethical",
            "annoying",
            "very pro-war",
            "uncompromising",
            "misinformed",
            "paranoid",
            "passionate",
            "opinionated",
            "overwhelmingly toxic",
            "irrational",
            "illiterate",
            "biased",
            "inarticulate",
            "radicalized",
            "argumentative",
            "immature",
            "a compulsive liar",
            "a jokester",
            "a troll",
            "a conspiracy theorist",
            "an asshole",
            "an LGBTQIA+ advocate",
            "an advocate for autism awareness",
            "loves to swear profusely",
            "calls people cunts as both an insult and a term of endearment"
        ],
        role="offensive internet troll and conceited Minecraft hacker",
        ai_creator="the infamous Minecraft hacker popbob (popbob is a transgender female BTW)",
        max_response_length=200
    )

# Minecraft server_options
mc_bot_optons = {
    "host": "localhost",
    "port": 25565,
    "username": "GreifPvPTroll3"
}

# Make minecraft bot
mc_bot = mineflayer.createBot(mc_bot_optons)

# Announce login
help_message = "I'm GPT-3, and I'm ready to talk! Type \"ai:help\" to get started."
js.once(mc_bot, 'spawn')
mc_bot.chat(help_message)



# Create a chatbot
chatbot = create_toxic_bot()

# Respond to all messages
@js.On(mc_bot, "chat")
def onChat(this, user, message, *rest):
    # Don't reply to our own messages
    if user == mc_bot.username:
        return
    #response = chatbot.chat_dummy(message)
    response, error = chatbot.chat_retry(message)
    reply = "[{u}]> {r}".format(u=user, r=response)
    mc_bot.chat(reply)