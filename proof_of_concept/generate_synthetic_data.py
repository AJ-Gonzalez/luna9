"""Generate synthetic conversation data for validation."""

def generate_code_review():
    """Generate a code review conversation (16 messages, 4x4 surface)."""
    messages = [
        "User: I'm reviewing this pull request that refactors our authentication middleware. Can you help me spot any potential issues?",

        "Assistant: I'd be happy to review it. First, let me check if the session token validation is still properly implemented. Does the refactored code maintain the same security guarantees?",

        "User: Yes, it uses JWT tokens with the same validation logic. But they moved the token refresh logic into a separate function.",

        "Assistant: That's good for separation of concerns. Make sure the refresh function properly validates the refresh token before issuing a new access token. Common mistake is trusting refresh tokens without verification.",

        "User: Right, I see it calls verify_refresh_token() first. What about rate limiting? The old middleware had rate limiting built in.",

        "Assistant: Check if rate limiting was extracted to a separate middleware layer. It's actually better practice to have rate limiting as its own concern rather than coupled with authentication.",

        "User: I don't see it in the auth middleware anymore. Should I ask them to add it back?",

        "Assistant: Not necessarily. Check if there's a separate rate-limiting middleware in the stack. Modern architectures often separate these concerns - authentication, rate limiting, and request validation as independent middleware layers.",

        "User: Good point. Switching topics - do you know anything about database connection pooling?",

        "Assistant: Yes, connection pooling is important for database performance. Are you seeing connection pool issues in your application?",

        "User: Our database connections keep timing out under load. We're using the default pool size of 10.",

        "Assistant: That's likely too small for high-traffic applications. A good starting point is around 20-50 connections depending on your concurrent request volume. But you'll need to monitor and tune based on actual usage patterns.",

        "User: Should we set it higher just to be safe?",

        "Assistant: Not necessarily - more isn't always better. Too many connections can overwhelm the database server. It's better to right-size based on metrics. Monitor your connection pool utilization and gradually increase if you're consistently hitting the limit.",

        "User: Makes sense. Back to the authentication PR - what about error handling?",

        "Assistant: Good question. Authentication middleware should handle errors carefully. Make sure it doesn't leak information about whether a user exists vs whether the password is wrong - both should return the same generic 'invalid credentials' message for security reasons."
    ]

    return messages

def generate_cooking():
    """Generate a recipe/cooking conversation (16 messages, 4x4 surface)."""
    messages = [
        "User: I want to make homemade pasta for the first time. What kind of flour should I use?",

        "Assistant: For traditional Italian pasta, you'll want tipo 00 flour. It's finely milled and creates a silky, smooth texture. You can also use all-purpose flour if that's what you have - it works fine, just slightly different texture.",

        "User: I only have all-purpose. Will that work okay?",

        "Assistant: Absolutely! All-purpose flour makes great fresh pasta. You'll use about 100 grams of flour per egg. The basic ratio is 1 egg to 100g flour, but you can adjust based on how the dough feels.",

        "User: How do I know when the dough is ready?",

        "Assistant: The dough should be smooth and elastic, not sticky. After kneading for about 10 minutes, it should spring back when you poke it. Let it rest for 30 minutes wrapped in plastic - this relaxes the gluten and makes rolling easier.",

        "User: Can I use a rolling pin or do I need a pasta machine?",

        "Assistant: You can definitely use a rolling pin! It takes more effort, but it's how pasta was made for centuries. Roll from the center outward, rotating the dough frequently. Aim for about 1/16 inch thickness - you should almost be able to see your hand through it.",

        "User: What shapes are easiest for beginners?",

        "Assistant: Fettuccine or pappardelle are great starter shapes - just cut the rolled dough into ribbons. Fettuccine is about 1/4 inch wide, pappardelle is wider at 3/4 to 1 inch. Farfalle (bow ties) are also fun and forgiving.",

        "User: Speaking of Italian food, what wine pairs well with pasta?",

        "Assistant: It depends on the sauce! Light tomato sauces pair beautifully with Chianti. Creamy sauces like carbonara go well with crisp white wines like Pinot Grigio. The key is matching the wine's weight to the sauce's richness.",

        "User: I'm making a simple tomato basil sauce.",

        "Assistant: Perfect! A medium-bodied Italian red like Chianti or Sangiovese would be lovely. The acidity in the wine complements the tomatoes, and the herbal notes work beautifully with the basil.",

        "User: How long does fresh pasta take to cook?",

        "Assistant: Much faster than dried pasta - usually just 2-3 minutes in boiling salted water. Fresh pasta is done when it floats to the surface and has a tender bite. Start checking at 2 minutes.",

        "User: Should I add oil to the cooking water?",

        "Assistant: That's a common myth actually - you don't need oil in the water. It can actually prevent sauce from adhering to the pasta. Just use plenty of salted water (should taste like the sea) and stir occasionally to prevent sticking.",

        "User: How much salt are we talking?",

        "Assistant: About 1-2 tablespoons of salt per 4-6 quarts of water. It seems like a lot, but most of it stays in the water. This is your only chance to season the pasta itself from the inside.",

        "User: Got it. One more question - can I make the dough ahead of time?",

        "Assistant: Yes! Wrapped tightly in plastic wrap, pasta dough keeps in the fridge for up to 24 hours. Actually, some people think it's better the next day because the gluten has more time to relax. You can also freeze cut pasta for up to 2 months."
    ]

    return messages

def generate_science():
    """Generate a scientific concepts conversation (16 messages, 4x4 surface)."""
    messages = [
        "User: Can you explain how photosynthesis actually works? I know plants convert sunlight to energy, but what's happening at a molecular level?",

        "Assistant: Photosynthesis happens in two main stages: the light-dependent reactions and the Calvin cycle. In the light reactions, chlorophyll absorbs photons and uses that energy to split water molecules, releasing oxygen and creating energy carriers (ATP and NADPH).",

        "User: So the oxygen we breathe is a byproduct of splitting water?",

        "Assistant: Exactly! The water molecule (H2O) gets split into oxygen (O2), which is released, and hydrogen ions and electrons, which are used to create those energy carriers I mentioned. This happens in the thylakoid membranes inside chloroplasts.",

        "User: What happens to the ATP and NADPH?",

        "Assistant: They power the Calvin cycle, which happens in the stroma of the chloroplast. The Calvin cycle uses the energy from ATP and electrons from NADPH to convert carbon dioxide from the air into glucose. This is where the actual 'food' (sugar) gets made.",

        "User: How many carbon dioxide molecules does it take to make one glucose?",

        "Assistant: Six CO2 molecules are needed to produce one glucose molecule (C6H12O6). The Calvin cycle has to turn six times, fixing one CO2 each time, to build up enough carbon to make one glucose. It's remarkably efficient.",

        "User: What limits how fast a plant can photosynthesize?",

        "Assistant: Several factors act as limiting factors: light intensity, CO2 concentration, temperature, and water availability. Whichever factor is in shortest supply becomes the bottleneck. This is called the law of limiting factors or Liebig's law.",

        "User: Could we increase crop yields by giving plants more CO2?",

        "Assistant: Actually yes - this is already done in some greenhouses. Enriching the atmosphere to 800-1000 ppm CO2 (vs the natural ~420 ppm) can increase photosynthesis rates significantly. But it only works if the other factors aren't limiting - you need adequate light, water, and nutrients.",

        "User: Is this related to why some people worry about climate change affecting plants?",

        "Assistant: Interestingly, it's complex. Rising CO2 can boost photosynthesis (CO2 fertilization effect), but climate change also brings stress from heat, drought, and changing rainfall patterns. The net effect varies by plant species and region - some might benefit initially, others will suffer.",

        "User: Do all plants photosynthesize the same way?",

        "Assistant: No! There are actually three main types: C3, C4, and CAM photosynthesis. C3 is the standard pathway I described. C4 plants like corn have evolved a more efficient mechanism for hot, sunny conditions. CAM plants like cacti open their stomata at night to conserve water.",

        "User: What makes C4 more efficient?",

        "Assistant: C4 plants spatially separate the initial CO2 fixation from the Calvin cycle using specialized bundle sheath cells. This concentrates CO2 around the Calvin cycle enzymes, reducing photorespiration - a wasteful process where the enzyme RuBisCO accidentally binds oxygen instead of CO2.",

        "User: So photorespiration is like an error in the process?",

        "Assistant: Sort of - it's an evolutionary relic. RuBisCO evolved when Earth's atmosphere had almost no oxygen, so it wasn't optimized to distinguish between CO2 and O2. In hot, dry conditions when plants close their stomata, oxygen builds up inside leaves and photorespiration becomes a real problem.",

        "User: Could we genetically engineer crops to avoid this?",

        "Assistant: Scientists are actively working on this! Some are trying to introduce C4 pathways into C3 crops like rice and wheat. Others are engineering more selective versions of RuBisCO. If successful, it could significantly boost crop yields, especially in warmer climates.",

        "User: That sounds promising for food security.",

        "Assistant: It is, though it's technically challenging and will take years to implement safely. The complexity of photosynthesis means changing one part can have unexpected effects elsewhere in the plant's metabolism. But the potential benefits make it worth pursuing carefully."
    ]

    return messages

def generate_travel():
    """Generate a travel planning conversation (16 messages, 4x4 surface)."""
    messages = [
        "User: I'm planning a two-week trip to Japan in spring. Should I try to see multiple cities or stay in one region?",

        "Assistant: For two weeks, you could comfortably see 2-3 main cities with some day trips. A classic route is Tokyo (5-6 days), Kyoto (4-5 days), and maybe Osaka (2-3 days). This gives you modern Japan, traditional culture, and great food without feeling too rushed.",

        "User: Is spring a good time to visit?",

        "Assistant: Spring is beautiful, especially late March to early April during cherry blossom season. But it's also peak tourist season, so expect crowds and higher prices. Book accommodation well in advance. Late April to May is lovely too - fewer crowds and pleasant weather.",

        "User: What should I book in advance versus leaving flexible?",

        "Assistant: Definitely book: accommodation, Japan Rail Pass (if using), and any special experiences like teamLab Borderless or famous restaurants. Leave flexible: most temples/shrines, wandering neighborhoods, smaller restaurants. Some of the best experiences come from spontaneous exploration.",

        "User: Is the Japan Rail Pass worth it for that itinerary?",

        "Assistant: For Tokyo-Kyoto-Osaka, it's borderline. A 7-day pass costs about ¥29,000, while individual shinkansen tickets Tokyo-Kyoto and back to Tokyo would be around ¥26,000. It becomes worth it if you're doing additional day trips like to Nara, Hiroshima, or Mount Fuji.",

        "User: What are some good day trips from Kyoto?",

        "Assistant: Nara is fantastic - about 45 minutes away, famous for temples and friendly deer. Osaka is close for street food and urban energy. Arashiyama bamboo grove is gorgeous. Fushimi Inari shrine with its thousands of orange torii gates is a must - go early morning to avoid crowds.",

        "User: I've heard Japanese is difficult for non-speakers. Should I be worried?",

        "Assistant: In major tourist areas, you'll be fine with English and translation apps. Many signs have English, and younger Japanese often speak some English. Learn a few basic phrases (thank you, excuse me) - locals really appreciate the effort. Google Translate's camera feature is incredibly helpful for menus.",

        "User: Speaking of travel logistics, what's the best way to get travel insurance?",

        "Assistant: I recommend comprehensive travel insurance that covers medical emergencies, trip cancellation, and lost luggage. Companies like World Nomads or SafetyWing are popular with travelers. Make sure it covers any activities you plan to do - skiing, scuba diving, etc. might need additional coverage.",

        "User: Do I need special coverage for Japan specifically?",

        "Assistant: Japan is generally very safe, but medical care can be expensive for foreigners without insurance. Make sure your policy covers emergency medical treatment and evacuation. Also check if your regular health insurance provides any international coverage - sometimes credit cards include basic travel insurance too.",

        "User: Good point. Back to the itinerary - what should I prioritize in Tokyo?",

        "Assistant: Mix the traditional and modern: Senso-ji temple in Asakusa, Meiji Shrine in Harajuku, teamLab digital art museum, Tsukiji Outer Market for food, Shibuya crossing, and neighborhood exploring in Shimokitazawa or Yanaka. Don't over-schedule - leave time to wander and discover.",

        "User: What about food? Any must-try dishes?",

        "Assistant: Beyond sushi and ramen, try: yakitori (grilled chicken skewers), okonomiyaki (savory pancake), tonkatsu (breaded pork cutlet), and kaiseki (traditional multi-course meal). In Osaka, definitely try their takoyaki and okonomiyaki - they're famous for it. Convenience store food is surprisingly good too!",

        "User: Are there cultural rules I should know for restaurants?",

        "Assistant: A few key ones: don't tip (it's considered rude), slurp your noodles (it's polite!), say 'itadakimasu' before eating, don't stick chopsticks upright in rice, and bring cash - many smaller places don't take cards. At sushi restaurants, it's okay to eat with hands or chopsticks, both are proper.",

        "User: How much cash should I carry?",

        "Assistant: Japan is still quite cash-based despite being high-tech. I'd carry ¥10,000-20,000 daily (about $70-140 USD). 7-Eleven ATMs are foreigner-friendly for withdrawals. Many restaurants, temples, and small shops only take cash. Bigger stores and hotels usually accept cards.",

        "User: This is super helpful! Any other general tips?",

        "Assistant: A few more: get a Suica or Pasmo card for trains/buses, it makes travel so much easier. Walking is often the best way to see neighborhoods. Japanese toilets are amazing - embrace the bidet feature! Most importantly, be respectful, patient, and open to experiences. Japan rewards curiosity and courtesy."
    ]

    return messages

def generate_book_analysis():
    """Generate a book/movie analysis conversation (12 messages, 3x4 surface)."""
    messages = [
        "User: I just finished '1984' by Orwell. The ending was devastating. Do you think Winston really loved Big Brother at the end, or was he just broken?",

        "Assistant: That's the horror of it - both are true simultaneously. Winston's final love for Big Brother is genuine, but only because the torture broke him completely. The Party doesn't just force compliance; it reconstructs your innermost feelings. Orwell is showing that totalitarianism's ultimate victory is making the oppressed love their oppression.",

        "User: It reminded me of Stockholm syndrome, but more complete.",

        "Assistant: Exactly, though it goes deeper than Stockholm syndrome. The Party doesn't just make Winston dependent on them - they literally rewrite his capacity for love itself. That's why Julia's fate is equally tragic. They didn't just split them up; they destroyed their ability to love each other. 'They got me a long time ago' - chilling line.",

        "User: The whole 'War is Peace' thing - are there modern parallels?",

        "Assistant: Absolutely. Doublethink exists whenever we're asked to hold contradictory beliefs without acknowledging the contradiction. Some see it in political rhetoric where the same action is praised or condemned based purely on who's doing it, or in corporate speak that frames layoffs as 'rightsizing.' The specific ideology matters less than the mechanism of control.",

        "User: What about the surveillance state aspect? With our smartphones and data tracking?",

        "Assistant: It's both more and less dystopian than 1984. We have more surveillance than Orwell imagined, but it's often voluntary and decentralized rather than state-mandated. The telescreens were government-installed; we bought our surveillance devices willingly. Some argue this makes it more insidious - self-imposed panopticons. Though the goals differ from the Party's total control... mostly.",

        "User: Do you think Orwell was warning about a specific political system?",

        "Assistant: Orwell was a democratic socialist writing against both Stalinist communism AND the fascism he'd fought in Spain. He wasn't warning about left vs right, but about any system that values power over truth. The Party's ideology is deliberately vague - 'Ingsoc' could be anything. The warning is about totalitarian methods, not specific political content.",

        "User: The appendix about Newspeak suggests that Ingsoc eventually fell. Do you read it that way?",

        "Assistant: That's a brilliant catch that many readers miss! The appendix is written in past tense, in standard English, discussing Newspeak as a historical curiosity. It implies someone, somewhere, eventually defeats the Party. It's Orwell's hidden thread of hope in an otherwise hopeless novel - totalitarianism contains the seeds of its own destruction.",

        "User: Speaking of dystopian fiction, how does 1984 compare to Brave New World?",

        "Assistant: They're opposite approaches to control. Orwell feared a society controlled by pain and punishment. Huxley feared one controlled by pleasure and distraction. One controls through deprivation, the other through overabundance. Both end up in the same place: populations unable to think critically or resist. Different analysts argue about which vision is more prescient for our current society.",

        "User: I'd say we're getting both - surveillance AND distraction.",

        "Assistant: That's a sharp observation. We have elements of both: social credit systems and data tracking (Orwellian), alongside social media dopamine loops and endless entertainment (Huxleyan). Amusing Ourselves to Death by Neil Postman argues that Huxley was more right - we're not oppressed by what we hate, but distracted by what we love. Though maybe the synthesis of both is the real danger.",

        "User: What about the role of O'Brien? Why did he pretend to be part of the resistance?",

        "Assistant: O'Brien represents the ultimate corruption of mentorship and trust. He didn't just arrest Winston; he personally cultivated their relationship to make the betrayal more devastating. It's the Party's sadism refined to art - they could have arrested Winston anytime, but breaking him required first becoming his confessor, his intellectual equal, his hope. The relationship made Room 101 possible.",

        "User: That's so much more cruel than just arresting him outright.",

        "Assistant: Exactly. The cruelty is the point. The Party doesn't want martyrs or external enemies - they want broken believers. O'Brien's year-long charade wasn't inefficient; it was necessary for the kind of total victory the Party demands. 'We shall squeeze you empty, and then we shall fill you with ourselves' - that requires first creating something worth emptying.",

        "User: I need to let this book sit for a while. It's a lot to process.",

        "Assistant: That's the sign of great literature - it doesn't just tell you a story, it changes how you see things. 1984 tends to rewire your perception of language, truth, and power. Take your time with it. And maybe read something lighter next - you've earned it after that ending!"
    ]

    return messages

def save_conversations():
    """Save all synthetic conversations to files."""
    import os
    from pathlib import Path

    # Use relative path from script location
    data_dir = Path(__file__).parent / "data"
    os.makedirs(data_dir, exist_ok=True)

    conversations = {
        "synthetic_code_review.txt": generate_code_review(),
        "synthetic_cooking.txt": generate_cooking(),
        "synthetic_science.txt": generate_science(),
        "synthetic_travel.txt": generate_travel(),
        "synthetic_book_analysis.txt": generate_book_analysis()
    }

    for filename, messages in conversations.items():
        filepath = data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(messages))
        print(f"Created: {filename} ({len(messages)} messages)")

if __name__ == "__main__":
    save_conversations()
    print("\nAll synthetic conversations generated successfully!")
