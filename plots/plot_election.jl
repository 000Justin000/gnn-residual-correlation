using Plots;
using CSV;
using DataFrames;
using VegaLite;
using VegaDatasets;

VOT = CSV.read("datasets/election/election.csv")
dat = Dict(zip(VOT[:fips_code], (VOT[:gop_2016]-VOT[:dem_2016])./(VOT[:gop_2016]+VOT[:dem_2016])))

us10m = dataset("us-10m");
df = DataFrame(Dict("id"=>collect(keys(dat)), "value"=>(x->min(+1.0,max(-1.0,x))).(collect(values(dat)))));

h = @vlplot(
        width=1500, height=1100,
        mark={:geoshape, stroke=:gray, strokeWidth=0.1},
        data={values=us10m, format={type=:topojson, feature=:counties}},
        transform=[{lookup=:id, from={data=df, key=:id, fields=["value"]}}],
        projection={type=:albersUsa},
        color={"value:q", scale={domain=[-1.0,+1.0], scheme=:redblue, reverse=true}, legend=nothing},
        config={view={stroke=nothing},axis={grid=false}}
    );

save("figs/election_2016_gt.svg", h);
