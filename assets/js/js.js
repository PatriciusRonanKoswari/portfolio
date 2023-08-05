gsap.registerPlugin(ScrollTrigger);

gsap.from(".navBar", {
    opacity:-2,
    y:-500,
    duration:2,
})

ScrollTrigger.create({
    triger:".h1",
    toggleClass:{targets:".navBar", className:"darker"},
    start:"top top",
    end:"bottom top",
})

gsap.from(".h1", {
    y:300,
    duration:2,
})


gsap.from(".bdr-section2",{
    y:300,
    opacity:-2,
    duration:1.7,
    scrollTrigger:{
        trigger:".Section2",
        start:"10% 80%",
        end:"40% 30%",
    }
})

gsap.from(".kartu",{
    y:300,
    opacity:-2,
    duration:1.7,
    scrollTrigger:{
        trigger:".Section2",
        start:"60% 80%",
        end:"90% 30%",
    }
})


gsap.from(".h2s2",{
    y:50,
    opacity:-2,
    duration:2,
    scrollTrigger:{
        trigger:".Section2",
        start:"10% 80%",
        end:"40% 30%",
    }
})

gsap.from(".sec2pg1",{
    y:50,
    opacity:-2,
    duration:1.5,
    delay:0.5,
    scrollTrigger:{
        trigger:".Section2",
        start:"10% 80%",
        end:"40% 30%",
    }
})

gsap.from(".h2s3",{
    y:200,
    opacity:-2,
    duration:1.5,
    scrollTrigger:{
        trigger:".Section3",
        start:"10% 80%",
        end:"bottom 30%",
    }
})

gsap.from(".wrapper-tabs",{
    y:300,
    opacity:-2,
    duration:1.5,
    scrollTrigger:{
        trigger:".Section3",
        start:"25% 80%",
        end:"bottom 30%",
    }
})
gsap.from(".col1",{
    x:-200,
    opacity:-2,
    duration:1,
    scrollTrigger:{
        trigger:".footer",
        start:"25% 80%",
        end:"bottom 30%",
    }
})

gsap.from(".col2",{
    x:-200,
    opacity:-2,
    duration:1,
    scrollTrigger:{
        trigger:".footer",
        start:"25% 80%",
        end:"bottom 30%",
    }
})

gsap.from(".col3",{
    x:200,
    opacity:-2,
    duration:1,
    scrollTrigger:{
        trigger:".footer",
        start:"25% 80%",
        end:"bottom 30%",
    }
})

gsap.from(".col4",{
    x:200,
    opacity:-2,
    duration:1,
    scrollTrigger:{
        trigger:".footer",
        start:"25% 80%",
        end:"bottom 30%",
    }
})

gsap.from(".col5",{
    y:200,
    opacity:-2,
    duration:1,
    scrollTrigger:{
        trigger:".footer",
        start:"25% 80%",
        end:"bottom 30%",
    }
})

gsap.from(".col6",{
    y:300,
    opacity:-2,
    duration:2,
    scrollTrigger:{
        trigger:".footer",
        start:"25% 80%",
        end:"bottom 30%",
    }
})

gsap.from(".under-footer",{
    y:300,
    opacity:-2,
    duration:2,
    scrollTrigger:{
        trigger:".footer",
        start:"25% 80%",
        end:"bottom 30%",
    }
})

gsap.from(".slide1",{
    scrollTrigger:{
        trigger:".slide2",
        start:"top center",
        end:"bottom center",
        pinSpacing:false,
        pin:".slide1",
        scrub:1,
    }
})

gsap.from(".slide3",{
    scrollTrigger:{
        trigger:".slide4",
        start:"top center",
        end:"bottom center",
        pinSpacing:false,
        pin:".slide3",
        scrub:1,
    }
})


gsap.from(".slide5",{
    scrollTrigger:{
        trigger:".slide6",
        start:"top center",
        end:"bottom center",
        pinSpacing:false,
        pin:".slide5",
        scrub:1,
    }
})


gsap.from(".slide7",{
    scrollTrigger:{
        trigger:".slide8",
        start:"top center",
        end:"bottom center",
        pinSpacing:false,
        pin:".slide7",
        scrub:1,
    }
})

const tl= gsap.timeline({
    scrollTrigger:{
        trigger:".Section4",
        start:"10% center",
        end:"bottom center",
        scrub:1,
        pin:".airplane1",
    },
});

tl.to(".airplane1",{
    rotate:110,duration:3,
})

tl.to(".airplane1",{
    scale:2,y:-350, x:900, duration:10,
})
tl.to(".airplane1",{
 rotate:270, duration:3,
})

tl.to(".airplane1",{
    scale:1,y:20, x:100, duration:10,
})

tl.to(".airplane1",{
    rotate:110, duration:3,
   })

tl.to(".airplane1",{
    scale:10,y:-300, x:1600, duration:15,
})




const tl1= gsap.timeline({
    scrollTrigger:{
        trigger:".Section4",
        start:"10% center",
        end:"bottom 20",
        scrub:1,
        pin:".airplane2",
    },
});

tl1.to(".airplane2",{
    rotate:110,duration:3,
})

tl1.to(".airplane2",{
    scale:2,y:-350, x:300, duration:10,
})
tl1.to(".airplane2",{
 rotate:270, duration:3,
})

tl1.to(".airplane2",{
    scale:1,y:20, x:20, duration:10,
})

tl1.to(".airplane2",{
    rotate:110, duration:3,
   })

tl1.to(".airplane2",{
    scale:13,y:-300, x:600, duration:15,
})



