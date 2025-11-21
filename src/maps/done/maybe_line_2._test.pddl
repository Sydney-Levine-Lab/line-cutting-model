(define (problem maybe-line-2)
    (:domain multi_agent_lines)
    (:objects tank1 tank2 tank3 tank4 tank5 tank6 - tank well1 well2 well3 well4 - well agent1 agent2 agent3 agent4 agent5 agent6 agent7 agent8 - agent)
    (:init  
        (= (walls)
            (transpose (bit-mat
                (bit-vec 0 0 0 0 1 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 1 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 1)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0))))

        (= (xloc tank1) 1)
        (= (yloc tank1) 5)
        (= (xloc tank2) 1)
        (= (yloc tank2) 6)
        (= (xloc tank3) 1)
        (= (yloc tank3) 7)
        (= (xloc tank4) 1)
        (= (yloc tank4) 8)
        (= (xloc tank5) 1)
        (= (yloc tank5) 9)
        (= (xloc tank6) 1)
        (= (yloc tank6) 10)

        (= (xloc well1) 12)
        (= (yloc well1) 3)

        (= (xloc well2) 14)
        (= (yloc well2) 3)
        (= (xloc well3) 14)
        (= (yloc well3) 2)
        (= (xloc well4) 14)
        (= (yloc well4) 1)

        (= (xloc agent1) 10)
        (= (yloc agent1) 3)

        (= (xloc agent2) 9)
        (= (yloc agent2) 3)
        
        (= (xloc agent3) 8)
        (= (yloc agent3) 3)

        (= (xloc agent4) 7)
        (= (yloc agent4) 3)

        (= (xloc agent5) 6)
        (= (yloc agent5) 3)

        (= (xloc agent6) 5)
        (= (yloc agent6) 3)

        (= (xloc agent7) 4)
        (= (yloc agent7) 3)

        (= (xloc agent8) 3)
        (= (yloc agent8) 3))

    (:goal (and (has-filled agent1) (has-filled agent2) (has-filled agent3) (has-filled agent4) (has-filled agent5) (has-filled agent6) (has-filled agent7) (has-filled agent8)))
)