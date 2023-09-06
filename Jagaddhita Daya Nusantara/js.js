gsap.registerPlugin(ScrollTrigger);

gsap.from(".section3-bg",{
    y:-600,
    scrollTrigger:{
        trigger:".Section3",
        start:"top center",
        end:"center top",
        scrub:1,
    }
})

gsap.from(".contact-absolute",{
    y:600,
    duration:2,
    scrollTrigger:{
        trigger:".wrapper-map",
        start:"center bottom",
        end:"center bottom",
    }
})



