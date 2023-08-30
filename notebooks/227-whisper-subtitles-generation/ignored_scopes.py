from nncf import IgnoredScope

ignored_scope1 = IgnoredScope(
     names=[
         "__module.blocks.0.mlp.2/aten::linear/MatMul_133",
         "__module.blocks.3.attn.query/aten::linear/MatMul_368",
         "__module.blocks.3.attn.key/aten::linear/MatMul_369",
         "__module.blocks.3.attn.value/aten::linear/MatMul_370",
         "__module.blocks.5.attn/aten::matmul/MatMul",
     ]
 )

ignored_scope2 = IgnoredScope(
     names=[
         "__module.blocks.3.attn.value/aten::linear/MatMul_370",
         "__module.blocks.3.attn.query/aten::linear/MatMul_368",
         "__module.blocks.3.attn.key/aten::linear/MatMul_369",
         "__module.blocks.0.mlp.0/aten::linear/MatMul_132",
         "__module.blocks.5.mlp.0/aten::linear/MatMul_707",
         "__module.blocks.5.attn.key/aten::linear/MatMul_599",
         "__module.blocks.5.attn.query/aten::linear/MatMul_598",
         "__module.blocks.5.attn.value/aten::linear/MatMul_600",
         "__module.blocks.4.attn.value/aten::linear/MatMul_485",
         "__module.blocks.4.attn.key/aten::linear/MatMul_484",
         "__module.blocks.4.attn.query/aten::linear/MatMul_483",
     ]
 )

ignored_scope3 = IgnoredScope(
     names=[
         "__module.blocks.3.attn.value/aten::linear/MatMul_370",
         "__module.blocks.3.attn.query/aten::linear/MatMul_368",
         "__module.blocks.3.attn.key/aten::linear/MatMul_369",
         "__module.blocks.5.attn.key/aten::linear/MatMul_599",
         "__module.blocks.5.attn.query/aten::linear/MatMul_598",
         "__module.blocks.5.attn.value/aten::linear/MatMul_600",
         "__module.blocks.5.mlp.0/aten::linear/MatMul_707",
         "__module.blocks.1.cross_attn.query/aten::linear/MatMul_199",
         "__module.blocks.5.cross_attn/aten::matmul/MatMul",
         "__module.blocks.4.cross_attn.out/aten::linear/MatMul_587",
         "__module.blocks.4.mlp.0/aten::linear/MatMul_592",
     ]
 )


ignored_scope4 = IgnoredScope(
    names=[
        "__module.blocks.5.attn.out/aten::linear/MatMul_690"
    ]
)


ignored_scope5 = IgnoredScope(
    names=[
        "__module.blocks.0.cross_attn/aten::matmul/MatMul",
        "__module.blocks.2.attn.value/aten::linear/MatMul_267",
        "__module.blocks.2.attn.query/aten::linear/MatMul_265",
        "__module.blocks.2.attn.key/aten::linear/MatMul_266"
    ]
)
