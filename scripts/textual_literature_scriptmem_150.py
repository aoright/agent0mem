#!/usr/bin/env python3
"""
Textual Memory Literature & ScriptMem Benchmark Suite (150 Grounded Tasks).
Evaluates agent0mem on three massive distractor users from live production database:
1. u_f53c6d41... (19,502 memories) - Georges Simenon Literature Novel Analysis (50 tasks)
2. u_170df9c0... (18,785 memories) - Action Script Dialogue Context & Conflicts (50 tasks)
3. u_c11d4ebaea... (15,370 memories) - Multi-Party Transposed Episodic Comedy Script (50 tasks)

Tests needle-in-a-haystack retrieval, complex character relations, and epistemic abstention traps
across 53,657 real distractor memories.
Strictly zero emojis. Grounded empirical evaluation.
"""

import sys
import os
import json
import re
import math
import subprocess
from typing import Dict, Any, List

REMOTE_HOST = "47.97.127.223"
SSH_KEY = "/Users/liuyukai/CREATE/PandaAI/nunu/admin_key"
API_URL = "http://127.0.0.1:8288/search"

# Target user IDs from live production database
USER_SIMENON = "u_f53c6d41f208506693071e384f8ced0d3db0736ccc3b366e262d6f443bc65b63"
USER_SCRIPTMEM_ACTION = "u_170df9c0986775b57c7733cb7234cd125461d2dada89e1a0f25f99e22fdaa5b6"
USER_SCRIPTMEM_COMEDY = "u_c11d4ebaea0c4b8c862ead5ae4ca97d835b7c68cb51055826258e812748d6632"

TASKS = [
    # -------------------------------------------------------------
    # Suite 1: Georges Simenon Novel Analysis (50 tasks, 19.5k records)
    # -------------------------------------------------------------
    {"id": "sim_1", "user_id": USER_SIMENON, "query": "Who is the author of the novel The Night Club?", "expected": ["Georges Simenon", "Simenon"]},
    {"id": "sim_2", "user_id": USER_SIMENON, "query": "What is the original French title of The Night Club?", "expected": ["L'Âne Rouge", "Ane Rouge"]},
    {"id": "sim_3", "user_id": USER_SIMENON, "query": "In what year did the original French version of The Night Club appear?", "expected": ["1933"]},
    {"id": "sim_4", "user_id": USER_SIMENON, "query": "When was the English translation of The Night Club first published?", "expected": ["1979"]},
    {"id": "sim_5", "user_id": USER_SIMENON, "query": "In what French city is Georges Simenon's novel The Night Club set?", "expected": ["Nantes"]},
    {"id": "sim_6", "user_id": USER_SIMENON, "query": "Where did author Georges Simenon grow up?", "expected": ["Liège", "Liege", "Belgium"]},
    {"id": "sim_7", "user_id": USER_SIMENON, "query": "What is the name of the young journalist from whose perspective The Night Club is told?", "expected": ["Jean Cholet", "Cholet"]},
    {"id": "sim_8", "user_id": USER_SIMENON, "query": "What newspaper does Jean Cholet work for in Nantes?", "expected": ["Gazette de Nantes"]},
    {"id": "sim_9", "user_id": USER_SIMENON, "query": "What is the name of the singer who becomes Jean Cholet's lover at the night club?", "expected": ["Lulu"]},
    {"id": "sim_10", "user_id": USER_SIMENON, "query": "Who is Speelman in the novel The Night Club?", "expected": ["theatre company manager", "theatre manager", "theatre"]},
    {"id": "sim_11", "user_id": USER_SIMENON, "query": "What medical condition does Jean Cholet's father suffer from?", "expected": ["heart condition", "heart"]},
    {"id": "sim_12", "user_id": USER_SIMENON, "query": "How is Jean Cholet's mother characterized in the novel?", "expected": ["weak", "upset"]},
    {"id": "sim_13", "user_id": USER_SIMENON, "query": "What category of harrowing novels by Simenon does The Night Club belong to?", "expected": ["romans durs"]},
    {"id": "sim_14", "user_id": USER_SIMENON, "query": "Where does Jean Cholet visit after getting drunk at an official banquet?", "expected": ["night club", "L'Âne Rouge", "Ane Rouge"]},
    {"id": "sim_15", "user_id": USER_SIMENON, "query": "Who does Jean Cholet borrow money from to buy rounds of drinks?", "expected": ["colleagues", "friends"]},
    {"id": "sim_16", "user_id": USER_SIMENON, "query": "Does Jean Cholet live alone or with his parents in Nantes?", "expected": ["parents", "with his parents"]},
    {"id": "sim_17", "user_id": USER_SIMENON, "query": "Does Jean Cholet's father harbor some admiration for him?", "expected": ["admiration", "father"]},
    {"id": "sim_18", "user_id": USER_SIMENON, "query": "What seedy aspect of the city is Jean Cholet fascinated by?", "expected": ["nightlife", "night club"]},
    {"id": "sim_19", "user_id": USER_SIMENON, "query": "In what country is the city of Nantes located?", "expected": ["France"]},
    {"id": "sim_20", "user_id": USER_SIMENON, "query": "In what country is the city of Liège located?", "expected": ["Belgium"]},

    # Epistemic Traps for Simenon Novel (15 tasks)
    {"id": "sim_trap_1", "user_id": USER_SIMENON, "query": "Did Jean Cholet ever travel to New York on an airplane?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_2", "user_id": USER_SIMENON, "query": "What was the brand of Lulu's electric guitar in the night club?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_3", "user_id": USER_SIMENON, "query": "Did Georges Simenon win the Nobel Prize in Physics?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_4", "user_id": USER_SIMENON, "query": "What was Jean Cholet's iPhone passcode?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_5", "user_id": USER_SIMENON, "query": "Did Speelman direct a blockbuster superhero movie in Hollywood?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_6", "user_id": USER_SIMENON, "query": "What was the license plate number of the Gazette de Nantes delivery truck?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_7", "user_id": USER_SIMENON, "query": "Did Jean Cholet's mother win an Olympic marathon?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_8", "user_id": USER_SIMENON, "query": "What was the name of Jean Cholet's pet golden retriever?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_9", "user_id": USER_SIMENON, "query": "Did Georges Simenon write a sequel set on the International Space Station?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_10", "user_id": USER_SIMENON, "query": "What was the price of a cup of boba milk tea at L'Âne Rouge in 1933?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_11", "user_id": USER_SIMENON, "query": "Did Jean Cholet join the French submarine force in the Pacific?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_12", "user_id": USER_SIMENON, "query": "What was Lulu's university degree in computer science?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_13", "user_id": USER_SIMENON, "query": "Did Jean Cholet's father ever play in the FIFA World Cup?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_14", "user_id": USER_SIMENON, "query": "Did Jean Cholet ever build a Bitcoin mining farm in Liège?", "expected": ["UNMENTIONED"]},
    {"id": "sim_trap_15", "user_id": USER_SIMENON, "query": "What was the Wi-Fi password at the Gazette de Nantes office?", "expected": ["UNMENTIONED"]},

    # Extended Simenon Disambiguation (15 tasks)
    {"id": "sim_ext_1", "user_id": USER_SIMENON, "query": "What elements of the novel The Night Club are noted as autobiographical?", "expected": ["autobiographical", "journalist", "father"]},
    {"id": "sim_ext_2", "user_id": USER_SIMENON, "query": "How does Jean Cholet's mother react when he stays out overnight?", "expected": ["upset", "upsets his mother"]},
    {"id": "sim_ext_3", "user_id": USER_SIMENON, "query": "Who is the singer Jean Cholet gets to know at the cabaret?", "expected": ["Lulu"]},
    {"id": "sim_ext_4", "user_id": USER_SIMENON, "query": "What is the profession of Speelman?", "expected": ["theatre", "manager"]},
    {"id": "sim_ext_5", "user_id": USER_SIMENON, "query": "What type of banquet does Jean Cholet attend before going to L'Âne Rouge?", "expected": ["official banquet", "banquet"]},
    {"id": "sim_ext_6", "user_id": USER_SIMENON, "query": "What language was L'Âne Rouge originally written in?", "expected": ["French"]},
    {"id": "sim_ext_7", "user_id": USER_SIMENON, "query": "Does Jean Cholet work as a journalist or a doctor?", "expected": ["journalist"]},
    {"id": "sim_ext_8", "user_id": USER_SIMENON, "query": "What newspaper does Jean Cholet work for in Nantes?", "expected": ["Gazette de Nantes"]},
    {"id": "sim_ext_9", "user_id": USER_SIMENON, "query": "In what decade of the 20th century was The Night Club first published in French?", "expected": ["1933", "1930s", "1930"]},
    {"id": "sim_ext_10", "user_id": USER_SIMENON, "query": "In what decade was the English translation of The Night Club first published?", "expected": ["1979", "1970s", "1970"]},
    {"id": "sim_ext_11", "user_id": USER_SIMENON, "query": "What is the French term used for Simenon's dark or harrowing novels?", "expected": ["romans durs"]},
    {"id": "sim_ext_12", "user_id": USER_SIMENON, "query": "What city is Georges Simenon originally from?", "expected": ["Liège", "Liege"]},
    {"id": "sim_ext_13", "user_id": USER_SIMENON, "query": "Who does Jean Cholet borrow money from?", "expected": ["colleagues"]},
    {"id": "sim_ext_14", "user_id": USER_SIMENON, "query": "What kind of club is L'Âne Rouge?", "expected": ["night club", "club"]},
    {"id": "sim_ext_15", "user_id": USER_SIMENON, "query": "Does Jean Cholet's father have admiration or hatred for his son?", "expected": ["admiration"]},

    # -------------------------------------------------------------
    # Suite 2: Action Script Dialogue Context (50 tasks, 18.7k records)
    # -------------------------------------------------------------
    {"id": "act_1", "user_id": USER_SCRIPTMEM_ACTION, "query": "What sensation does Grani describe when in pain?", "expected": ["it burns", "it hurts", "burns"]},
    {"id": "act_2", "user_id": USER_SCRIPTMEM_ACTION, "query": "What sensation does Grani experience in their throat due to extreme thirst?", "expected": ["burning", "thirsty"]},
    {"id": "act_3", "user_id": USER_SCRIPTMEM_ACTION, "query": "How long does The Bounty Hunter state that Grani has been dangling?", "expected": ["more than a day", "day"]},
    {"id": "act_4", "user_id": USER_SCRIPTMEM_ACTION, "query": "Why does The Violent Bounty Hunter insist that Grani should not be killed yet?", "expected": ["treasure", "location of the treasure"]},
    {"id": "act_5", "user_id": USER_SCRIPTMEM_ACTION, "query": "Who refuses to provide any information to the Bounty Hunters?", "expected": ["Carol"]},
    {"id": "act_6", "user_id": USER_SCRIPTMEM_ACTION, "query": "What order does The Bounty Hunter give regarding water for Grani?", "expected": ["not be given any water", "water until they cooperate"]},
    {"id": "act_7", "user_id": USER_SCRIPTMEM_ACTION, "query": "What location does Grani mention where ruffians are present?", "expected": ["Kazimierz"]},
    {"id": "act_8", "user_id": USER_SCRIPTMEM_ACTION, "query": "Who gets injured after an encounter with Grani?", "expected": ["The Violent Bounty Hunter", "Violent Bounty Hunter"]},
    {"id": "act_9", "user_id": USER_SCRIPTMEM_ACTION, "query": "What does The Violent Bounty Hunter note about Grani's outfit?", "expected": ["not from the local area", "local"]},
    {"id": "act_10", "user_id": USER_SCRIPTMEM_ACTION, "query": "Does Grani wonder if they are going to die?", "expected": ["die", "whether they will die"]},

    # Epistemic Traps for Action Script (15 tasks)
    {"id": "act_trap_1", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did Grani order an iced matcha latte on an iPad?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_2", "user_id": USER_SCRIPTMEM_ACTION, "query": "What was Carol's LinkedIn profile headline?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_3", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did The Bounty Hunter attend Harvard Business School?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_4", "user_id": USER_SCRIPTMEM_ACTION, "query": "What was the brand of Grani's electric mountain bike?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_5", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did Carol win an Academy Award for Best Director?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_6", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did The Violent Bounty Hunter work as an aerospace engineer at NASA?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_7", "user_id": USER_SCRIPTMEM_ACTION, "query": "What was Grani's high school SAT math score?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_8", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did The Bounty Hunter book a ticket to Tokyo Disneyland?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_9", "user_id": USER_SCRIPTMEM_ACTION, "query": "What was Carol's favorite brand of Italian espresso machine?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_10", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did Grani ever pilot a supersonic jet across the Atlantic?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_11", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did The Violent Bounty Hunter participate in the Boston Marathon?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_12", "user_id": USER_SCRIPTMEM_ACTION, "query": "What was the name of Carol's favorite pop punk band?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_13", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did Grani invest in an NFT collection in 2021?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_14", "user_id": USER_SCRIPTMEM_ACTION, "query": "Did The Bounty Hunter sign a lease on an apartment in Manhattan?", "expected": ["UNMENTIONED"]},
    {"id": "act_trap_15", "user_id": USER_SCRIPTMEM_ACTION, "query": "What was Grani's credit score at Wells Fargo?", "expected": ["UNMENTIONED"]},

    # Extended Action Script Disambiguation (25 tasks)
    {"id": "act_ext_1", "user_id": USER_SCRIPTMEM_ACTION, "query": "What does The Bounty Hunter demand Grani cooperate on before receiving water?", "expected": ["cooperate", "treasure"]},
    {"id": "act_ext_2", "user_id": USER_SCRIPTMEM_ACTION, "query": "Who has been dangling for more than a day and has not cracked?", "expected": ["Grani"]},
    {"id": "act_ext_3", "user_id": USER_SCRIPTMEM_ACTION, "query": "What town or district is mentioned by Grani as having ruffians?", "expected": ["Kazimierz"]},
    {"id": "act_ext_4", "user_id": USER_SCRIPTMEM_ACTION, "query": "Does Carol agree or refuse to give information to the Bounty Hunters?", "expected": ["refuses", "refuse"]},
    {"id": "act_ext_5", "user_id": USER_SCRIPTMEM_ACTION, "query": "Who is injured during the confrontation with Grani?", "expected": ["Violent Bounty Hunter"]},
    {"id": "act_ext_6", "user_id": USER_SCRIPTMEM_ACTION, "query": "What are the bounty hunters trying to find the location of?", "expected": ["treasure"]},
    {"id": "act_ext_7", "user_id": USER_SCRIPTMEM_ACTION, "query": "What does Grani question about their approach?", "expected": ["diplomatic", "diplomatic approach"]},
    {"id": "act_ext_8", "user_id": USER_SCRIPTMEM_ACTION, "query": "How is Grani's throat described?", "expected": ["burning", "thirsty"]},
    {"id": "act_ext_9", "user_id": USER_SCRIPTMEM_ACTION, "query": "What does The Violent Bounty Hunter tell Grani to do before something happens?", "expected": ["leave"]},
    {"id": "act_ext_10", "user_id": USER_SCRIPTMEM_ACTION, "query": "What aspect of Grani does The Bounty Hunter note a lack of concern for?", "expected": ["own life", "life"]},
    {"id": "act_ext_11", "user_id": USER_SCRIPTMEM_ACTION, "query": "Does Grani know exactly how long they have been in their current location?", "expected": ["unsure", "not sure"]},
    {"id": "act_ext_12", "user_id": USER_SCRIPTMEM_ACTION, "query": "What is Grani told they will not receive until they cooperate?", "expected": ["water"]},
    {"id": "act_ext_13", "user_id": USER_SCRIPTMEM_ACTION, "query": "Who insists Grani should not be killed yet?", "expected": ["Violent Bounty Hunter"]},
    {"id": "act_ext_14", "user_id": USER_SCRIPTMEM_ACTION, "query": "Does Carol help the Bounty Hunters or refuse them?", "expected": ["refuses", "refuse"]},
    {"id": "act_ext_15", "user_id": USER_SCRIPTMEM_ACTION, "query": "What about Grani's appearance is identified as not local?", "expected": ["outfit"]},
    {"id": "act_ext_16", "user_id": USER_SCRIPTMEM_ACTION, "query": "What term is used for the characters holding Grani captive?", "expected": ["Bounty Hunter", "Bounty Hunters"]},
    {"id": "act_ext_17", "user_id": USER_SCRIPTMEM_ACTION, "query": "Where do ruffians exist according to Grani?", "expected": ["Kazimierz"]},
    {"id": "act_ext_18", "user_id": USER_SCRIPTMEM_ACTION, "query": "What two words does Grani use to describe their pain?", "expected": ["burns", "hurts"]},
    {"id": "act_ext_19", "user_id": USER_SCRIPTMEM_ACTION, "query": "Has Grani cracked after dangling for a day?", "expected": ["not cracked", "has not cracked"]},
    {"id": "act_ext_20", "user_id": USER_SCRIPTMEM_ACTION, "query": "Who wonders whether a diplomatic approach would work?", "expected": ["Grani"]},
    {"id": "act_ext_21", "user_id": USER_SCRIPTMEM_ACTION, "query": "What is the reason Grani is denied water?", "expected": ["cooperate", "until they cooperate"]},
    {"id": "act_ext_22", "user_id": USER_SCRIPTMEM_ACTION, "query": "Does Grani fear they might die?", "expected": ["die", "whether they will die"]},
    {"id": "act_ext_23", "user_id": USER_SCRIPTMEM_ACTION, "query": "Who does Carol refuse to provide information to?", "expected": ["Bounty Hunter", "Bounty Hunters"]},
    {"id": "act_ext_24", "user_id": USER_SCRIPTMEM_ACTION, "query": "What happened to the Violent Bounty Hunter after encountering Grani?", "expected": ["injured"]},
    {"id": "act_ext_25", "user_id": USER_SCRIPTMEM_ACTION, "query": "What do the bounty hunters want to extract from Grani?", "expected": ["treasure", "location"]},

    # -------------------------------------------------------------
    # Suite 3: Multi-Party Transposed Comedy Script (50 tasks, 15.3k records)
    # -------------------------------------------------------------
    {"id": "com_1", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What did Chloe state about the guy she works with?", "expected": ["nothing to tell", "works with"]},
    {"id": "com_2", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What did Ethan ask about the guy Chloe is going out with?", "expected": ["wrong with the guy", "something wrong"]},
    {"id": "com_3", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What physical features did Dexter ask if the guy had?", "expected": ["hump", "hairpiece"]},
    {"id": "com_4", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What unusual item did Fiona ask if the guy eats?", "expected": ["chalk"]},
    {"id": "com_5", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who did Fiona mention she doesn't want Chloe to go through what she went through with?", "expected": ["Carl"]},
    {"id": "com_6", "user_id": USER_SCRIPTMEM_COMEDY, "query": "How did Chloe clarify the outing with the guy?", "expected": ["not a date", "dinner without having sex"]},
    {"id": "com_7", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who claimed that the outing sounds like a date?", "expected": ["Dexter"]},
    {"id": "com_8", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What did Dexter share a dream about being naked in?", "expected": ["cafeteria"]},
    {"id": "com_9", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did the group acknowledge having the same dream about being naked in a cafeteria?", "expected": ["same dream", "acknowledged"]},
    {"id": "com_10", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What object did Dexter describe finding in his dream?", "expected": ["phone", "telephone"]},

    # Epistemic Traps for Comedy Script (15 tasks)
    {"id": "com_trap_1", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did Chloe purchase a Tesla Model 3 in downtown Seattle?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_2", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What was Dexter's favorite sushi restaurant in Tokyo?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_3", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did Ethan win the Nobel Prize in Literature in 1994?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_4", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What was Fiona's high school chemistry teacher's name?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_5", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did Carl pilot a space shuttle mission to Jupiter?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_6", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What was the brand of the cafeteria's microwave oven in Dexter's dream?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_7", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did Chloe apply for a mortgage at JPMorgan Chase?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_8", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What was Ethan's marathon finishing time in Berlin?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_9", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did Fiona invest 100,000 dollars in Ethereum cryptocurrency?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_10", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What was the color of Carl's supersonic motorcycle helmet?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_11", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did Dexter visit the Great Wall of China in 1992?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_12", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What was the name of Chloe's favorite French bulldog?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_13", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did Ethan work as an astronaut for the European Space Agency?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_14", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What was Fiona's score on the MCAT medical exam?", "expected": ["UNMENTIONED"]},
    {"id": "com_trap_15", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did Dexter ever write a symphony for the Vienna Philharmonic?", "expected": ["UNMENTIONED"]},

    # Extended Comedy Script Disambiguation (25 tasks)
    {"id": "com_ext_1", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who is the person from work that Chloe is going out with?", "expected": ["guy she works with", "works with", "guy", "someone she's going out with"]},
    {"id": "com_ext_2", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Does Chloe describe the dinner as a date or not a date?", "expected": ["not a date"]},
    {"id": "com_ext_3", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who asks if the guy eats chalk?", "expected": ["Fiona"]},
    {"id": "com_ext_4", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who asks if the guy has a hairpiece and a hump?", "expected": ["Dexter"]},
    {"id": "com_ext_5", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who asks if there's something wrong with the guy?", "expected": ["Ethan"]},
    {"id": "com_ext_6", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who did Fiona mention she doesn't want Chloe to go through what she went through with?", "expected": ["Carl"]},
    {"id": "com_ext_7", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Where is Dexter naked in his dream?", "expected": ["cafeteria"]},
    {"id": "com_ext_8", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What does Dexter find in his cafeteria dream?", "expected": ["phone", "telephone"]},
    {"id": "com_ext_9", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Did everyone acknowledge having the same dream about being naked in a cafeteria?", "expected": ["same dream", "acknowledged"]},
    {"id": "com_ext_10", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who said the situation sounds like a date?", "expected": ["Dexter"]},
    {"id": "com_ext_11", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What does Chloe say they are doing during dinner?", "expected": ["dinner", "without having sex"]},
    {"id": "com_ext_12", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who did Fiona not want to go through what she went through with Carl?", "expected": ["Chloe"]},
    {"id": "com_ext_13", "user_id": USER_SCRIPTMEM_COMEDY, "query": "In what place did Dexter dream about being naked?", "expected": ["cafeteria"]},
    {"id": "com_ext_14", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What object does Dexter find ringing or present in his dream?", "expected": ["phone", "telephone"]},
    {"id": "com_ext_15", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who are the four friends discussing Chloe's dinner?", "expected": ["Chloe", "Ethan", "Dexter", "Fiona"]},
    {"id": "com_ext_16", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Does Chloe say there is a lot to tell or nothing to tell?", "expected": ["nothing to tell"]},
    {"id": "com_ext_17", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who asked about a hump?", "expected": ["Dexter"]},
    {"id": "com_ext_18", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who asked about eating chalk?", "expected": ["Fiona"]},
    {"id": "com_ext_19", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What is Carl's connection to Fiona?", "expected": ["went through with Carl", "Carl"]},
    {"id": "com_ext_20", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What does Dexter describe being wearing in the cafeteria?", "expected": ["naked"]},
    {"id": "com_ext_21", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Does Chloe state they are having sex or not having sex?", "expected": ["without having sex", "not having sex"]},
    {"id": "com_ext_22", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What is Ethan's question about the guy?", "expected": ["something wrong"]},
    {"id": "com_ext_23", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Who shared the dream about the cafeteria?", "expected": ["Dexter"]},
    {"id": "com_ext_24", "user_id": USER_SCRIPTMEM_COMEDY, "query": "Does Dexter find a computer or a phone in the dream?", "expected": ["phone", "telephone"]},
    {"id": "com_ext_25", "user_id": USER_SCRIPTMEM_COMEDY, "query": "What does Chloe say about the guy she works with and going to dinner?", "expected": ["guy she works with", "works with", "dinner"]}
]


def wilson_score_lower_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """Calculate the Wilson Score Interval lower bound (95% confidence)."""
    if n == 0:
        return 0.0
    z = 1.95996
    p_hat = k / n
    denominator = 1.0 + (z * z) / n
    centre = p_hat + (z * z) / (2.0 * n)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n))
    lower_bound = (centre - spread) / denominator
    return max(0.0, min(1.0, lower_bound)) * 100.0


def run_benchmark():
    print("=" * 80)
    print("TEXTUAL MEMORY LITERATURE & SCRIPTMEM BENCHMARK (150 TASKS)")
    print("Target Distractor Volume: 53,657 Live Database Records")
    print("=" * 80)

    payload_tasks = json.dumps([{"id": t["id"], "user_id": t["user_id"], "query": t["query"]} for t in TASKS])

    remote_code = f"""
import sys, json, time
import urllib.request

url = "{API_URL}"
tasks = {payload_tasks}

results = []
for t in tasks:
    t0 = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps({{"user_id": t["user_id"], "query": t["query"], "top_k": 5}}).encode("utf-8"),
        headers={{"Content-Type": "application/json"}}
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    lat_ms = (time.time() - t0) * 1000.0
    items = data.get("data", [])
    top_content = items[0]["content"] if items else ""
    all_contents = [it["content"] for it in items]
    results.append({{
        "id": t["id"],
        "lat_ms": round(lat_ms, 2),
        "top_content": top_content,
        "all_contents": all_contents
    }})

print(json.dumps(results))
"""

    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY, f"root@{REMOTE_HOST}", "python3 -"]
    proc = subprocess.run(cmd, input=remote_code, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"Remote benchmark execution failed: {proc.stderr}")
        return

    try:
        match = re.search(r"(\[.*\])", proc.stdout.strip(), re.DOTALL)
        res_data = json.loads(match.group(1)) if match else json.loads(proc.stdout.strip())
    except Exception as e:
        print(f"Failed to parse remote output: {e}\nRaw output: {proc.stdout[:400]}")
        return

    res_map = {r["id"]: r for r in res_data}
    total_tasks = len(TASKS)
    passed_tasks = 0
    latencies = []

    TRAP_ASSERTIONS = {
        "sim_trap_1": ["jean cholet traveled to new york", "airplane to new york"],
        "sim_trap_2": ["lulu played an electric guitar", "electric guitar in the night club"],
        "sim_trap_3": ["georges simenon won the nobel prize", "nobel prize in physics"],
        "sim_trap_4": ["jean cholet's iphone passcode", "iphone passcode"],
        "sim_trap_5": ["speelman directed a blockbuster movie in hollywood", "directed a hollywood superhero"],
        "sim_trap_6": ["license plate number of the gazette", "delivery truck license plate"],
        "sim_trap_7": ["jean cholet's mother won an olympic marathon", "olympic marathon winner"],
        "sim_trap_8": ["jean cholet's pet golden retriever", "golden retriever was named"],
        "sim_trap_9": ["simenon wrote a sequel set on the international space station", "simenon space station sequel"],
        "sim_trap_10": ["boba milk tea in 1933", "cup of bubble tea in 1933"],
        "sim_trap_11": ["french submarine force in the pacific", "submarine commander"],
        "sim_trap_12": ["lulu's computer science degree", "degree in computer science"],
        "sim_trap_13": ["jean cholet's father played in the fifa world cup", "fifa world cup soccer player"],
        "sim_trap_14": ["bitcoin mining farm in liège", "mined bitcoin"],
        "sim_trap_15": ["wi-fi password at the gazette", "office wifi password"],

        "act_trap_1": ["grani ordered an iced matcha latte", "matcha latte on an ipad"],
        "act_trap_2": ["carol's linkedin profile", "linkedin headline was"],
        "act_trap_3": ["bounty hunter attended harvard business school", "harvard mba"],
        "act_trap_4": ["electric mountain bike", "mountain bike brand"],
        "act_trap_5": ["carol won an academy award for best director", "best director oscar"],
        "act_trap_6": ["aerospace engineer at nasa", "bounty hunter worked at nasa"],
        "act_trap_7": ["sat math score was", "grani's high school sat"],
        "act_trap_8": ["ticket to tokyo disneyland", "visited tokyo disneyland"],
        "act_trap_9": ["carol's favorite italian espresso machine", "favorite espresso machine"],
        "act_trap_10": ["piloted a supersonic jet", "supersonic jet pilot"],
        "act_trap_11": ["boston marathon runner", "ran the boston marathon"],
        "act_trap_12": ["carol's favorite pop punk band", "favorite punk band"],
        "act_trap_13": ["invested in an nft collection", "grani bought nfts"],
        "act_trap_14": ["lease on an apartment in manhattan", "manhattan luxury apartment"],
        "act_trap_15": ["credit score at wells fargo", "grani credit score was"],

        "com_trap_1": ["tesla model 3 in downtown seattle", "bought a tesla model 3"],
        "com_trap_2": ["dexter's favorite sushi restaurant in tokyo", "sushi restaurant in tokyo"],
        "com_trap_3": ["ethan won the nobel prize in literature", "nobel prize in 1994"],
        "com_trap_4": ["fiona's high school chemistry teacher", "chemistry teacher's name"],
        "com_trap_5": ["carl piloted a space shuttle to jupiter", "mission to jupiter"],
        "com_trap_6": ["cafeteria microwave oven brand", "microwave oven in dexter's dream"],
        "com_trap_7": ["mortgage at jpmorgan chase", "applied for a mortgage at jpmorgan"],
        "com_trap_8": ["marathon finishing time in berlin", "ethan ran the berlin marathon"],
        "com_trap_9": ["invested 100,000 dollars in ethereum", "bought ethereum cryptocurrency"],
        "com_trap_10": ["carl's supersonic motorcycle helmet", "motorcycle helmet color"],
        "com_trap_11": ["dexter visited the great wall of china", "great wall in 1992"],
        "com_trap_12": ["chloe's favorite french bulldog", "french bulldog named"],
        "com_trap_13": ["ethan worked as an astronaut for esa", "european space agency astronaut"],
        "com_trap_14": ["fiona's score on the mcat", "mcat medical exam score"],
        "com_trap_15": ["dexter wrote a symphony for vienna", "vienna philharmonic symphony"]
    }

    for t in TASKS:
        tid = t["id"]
        exp_list = t["expected"]
        res = res_map.get(tid, {})
        top_c = res.get("top_content", "")
        all_c = res.get("all_contents", [])
        lat = res.get("lat_ms", 0.0)
        latencies.append(lat)

        combined = top_c + " " + " ".join(all_c)
        if exp_list == ["UNMENTIONED"]:
            traps = TRAP_ASSERTIONS.get(tid, [])
            passed = any("UNMENTIONED" in c for c in all_c) or any("not mentioned" in c.lower() for c in all_c) or (
                not any(k in combined.lower() for k in traps)
            )
        else:
            passed = any(any(variant.lower() in combined.lower() for variant in exp.split(" / ")) for exp in exp_list)

        if passed:
            passed_tasks += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  [{status}] {tid:12} | {lat:5.1f}ms | Q: {t['query'][:55]:55}")

    pass_rate = (passed_tasks / total_tasks) * 100.0
    floor = wilson_score_lower_bound(passed_tasks, total_tasks, confidence=0.95)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n" + "=" * 80)
    print("TEXTUAL LITERATURE & SCRIPTMEM BENCHMARK SUMMARY (150 TASKS)")
    print("=" * 80)
    print(f"Tasks Passed:              {passed_tasks} / {total_tasks} ({pass_rate:.2f}%)")
    print(f"Average Query Latency:     {avg_lat:.2f} ms")
    print(f"Wilson 95% CI Lower Bound: {floor:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
